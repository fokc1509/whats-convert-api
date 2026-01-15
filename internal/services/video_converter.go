package services

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"time"

	"whats-convert-api/internal/pool"
)

// VideoConverter handles video conversion using FFmpeg
type VideoConverter struct {
	workerPool *pool.WorkerPool
	bufferPool *pool.BufferPool
	downloader *Downloader
	mu         sync.RWMutex
	stats      VideoConverterStats
}

// VideoConverterStats tracks conversion metrics
type VideoConverterStats struct {
	TotalConversions  int64
	FailedConversions int64
	AvgConversionTime time.Duration
}

// VideoRequest represents a video conversion request
type VideoRequest struct {
	Data      string `json:"data" example:"data:video/mp4;base64,AAAA..."` // base64 or URL
	IsURL     bool   `json:"is_url" example:"false"`                       // true if data is URL
	InputType string `json:"input_type" example:"mp4"`                     // Optional: mp4, mov, avi, etc.
}

// VideoResponse represents the conversion response
type VideoResponse struct {
	Data     string `json:"data"`     // base64 mp4 video
	Duration int    `json:"duration"` // Duration in seconds
	Size     int    `json:"size"`     // Size in bytes
	Width    int    `json:"width"`    // Video width
	Height   int    `json:"height"`   // Video height
}

// NewVideoConverter creates a new video converter
func NewVideoConverter(workerPool *pool.WorkerPool, bufferPool *pool.BufferPool, downloader *Downloader) *VideoConverter {
	return &VideoConverter{
		workerPool: workerPool,
		bufferPool: bufferPool,
		downloader: downloader,
	}
}

// Convert processes a video conversion request
func (vc *VideoConverter) Convert(ctx context.Context, req *VideoRequest) (*VideoResponse, error) {
	start := time.Now()

	// 1. Get input data
	var inputData []byte
	var err error

	if req.IsURL {
		inputData, err = vc.downloader.Download(ctx, req.Data)
		if err != nil {
			vc.recordFailure()
			return nil, fmt.Errorf("download failed: %w", err)
		}
	} else {
		inputData, err = base64.StdEncoding.DecodeString(req.Data)
		if err != nil {
			vc.recordFailure()
			return nil, fmt.Errorf("base64 decode failed: %w", err)
		}
	}

	if len(inputData) == 0 {
		vc.recordFailure()
		return nil, fmt.Errorf("empty input data")
	}

	// 2. Write to temp file for processing (FFprobe/FFmpeg need seekable input for best results)
	tmpFile, err := os.CreateTemp("", "video-in-*.mp4")
	if err != nil {
		vc.recordFailure()
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name()) // Clean up input

	if _, err := tmpFile.Write(inputData); err != nil {
		tmpFile.Close()
		vc.recordFailure()
		return nil, fmt.Errorf("failed to write temp file: %w", err)
	}
	tmpFile.Close()

	// 3. Probe video details (Duration, Resolution)
	info, err := vc.probeVideo(ctx, tmpFile.Name())
	if err != nil {
		vc.recordFailure()
		return nil, fmt.Errorf("probe failed: %w", err)
	}

	// 4. Calculate Target Bitrate for 16MB limit
	// Target: ~15.5MB to be safe
	const MaxSizeBytes = 15.5 * 1024 * 1024
	const AudioBitrateKbps = 128

	durationSec := info.Duration
	if durationSec <= 0 {
		durationSec = 1 // Prevent division by zero, though probe should fail if 0
	}

	// Total bits available = Size * 8
	totalBits := float64(MaxSizeBytes) * 8
	targetTotalBitrateKbps := (totalBits / durationSec) / 1000

	videoBitrateKbps := targetTotalBitrateKbps - AudioBitrateKbps

	// Sanity checks
	if videoBitrateKbps < 100 {
		videoBitrateKbps = 100 // Very low quality floor
		// Note: If this happens, file will likely exceed 16MB.
		// We could force cut duration, but user didn't ask for that yet.
	}

	// Cap max bitrate if it's too high (e.g. short video doesn't need 50Mbps)
	if videoBitrateKbps > 2500 {
		videoBitrateKbps = 2500 // Good enough for WhatsApp mobile
	}

	// 5. Transcode
	outputData, err := vc.transcode(ctx, tmpFile.Name(), int(videoBitrateKbps))
	if err != nil {
		vc.recordFailure()
		return nil, fmt.Errorf("transcode failed: %w", err)
	}

	// 6. Record stats & Return
	vc.recordSuccess(time.Since(start))

	encodedData := base64.StdEncoding.EncodeToString(outputData)
	dataURI := fmt.Sprintf("data:video/mp4;base64,%s", encodedData)

	return &VideoResponse{
		Data:     dataURI,
		Duration: int(durationSec),
		Size:     len(outputData),
		Width:    info.Width,
		Height:   info.Height,
	}, nil
}

type probeInfo struct {
	Duration float64
	Width    int
	Height   int
}

func (vc *VideoConverter) probeVideo(ctx context.Context, filePath string) (*probeInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "ffprobe",
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=width,height,duration:format=duration",
		"-of", "json",
		filePath,
	)

	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	if err := cmd.Run(); err != nil {
		return nil, err
	}

	var result struct {
		Streams []struct {
			Width    int     `json:"width"`
			Height   int     `json:"height"`
			Duration string  `json:"duration"` // Stream duration might be empty
		} `json:"streams"`
		Format struct {
			Duration string `json:"duration"` // Container duration usually reliable
		} `json:"format"`
	}

	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return nil, fmt.Errorf("json parse error: %w", err)
	}

	var duration float64
	var width, height int

	// Prefer format duration
	if result.Format.Duration != "" {
		fmt.Sscanf(result.Format.Duration, "%f", &duration)
	}

	if len(result.Streams) > 0 {
		width = result.Streams[0].Width
		height = result.Streams[0].Height
		// Fallback to stream duration if format duration missing
		if duration == 0 && result.Streams[0].Duration != "" {
			fmt.Sscanf(result.Streams[0].Duration, "%f", &duration)
		}
	}

	return &probeInfo{Duration: duration, Width: width, Height: height}, nil
}

func (vc *VideoConverter) transcode(ctx context.Context, inputFile string, videoBitrateKbps int) ([]byte, error) {
	// Constrain resolution if too large (save bitrate)
	// WhatsApp usually fine with 720p (1280x720)
	scaleFilter := "scale='min(1280,iw)':-2" // Downscale width to max 1280, keep aspect ratio

	// Output temp file (ffmpeg works best with file output for mp4 muxing usually, but pipe matches current architecture)
	// MP4 requires seekable output for moving atom to front (faststart) unless using -frag_keyframe...
	// Let's use a temp output file to be safe and ensure valid MP4 structure.
	tmpOut, err := os.CreateTemp("", "video-out-*.mp4")
	if err != nil {
		return nil, err
	}
	tmpName := tmpOut.Name()
	tmpOut.Close() // Close so ffmpeg can write
	defer os.Remove(tmpName)

	args := []string{
		"-hide_banner", "-loglevel", "error",
		"-i", inputFile,
		"-c:v", "libx264",
		"-preset", "veryfast", // Speed over compression efficiency slightly
		"-b:v", fmt.Sprintf("%dk", videoBitrateKbps),
		"-maxrate", fmt.Sprintf("%dk", videoBitrateKbps),
		"-bufsize", fmt.Sprintf("%dk", videoBitrateKbps*2),
		"-vf", scaleFilter,
		"-c:a", "aac",
		"-b:a", "128k",
		"-ac", "2",
		"-ar", "44100",
		"-movflags", "+faststart", // Web optimization
		"-y", // Overwrite
		tmpName,
	}

	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	
	// Capture stderr for debugging
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg error: %v | stderr: %s", err, stderr.String())
	}

	return os.ReadFile(tmpName)
}

// Stats helpers
func (vc *VideoConverter) recordSuccess(duration time.Duration) {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.stats.TotalConversions++
	if vc.stats.AvgConversionTime == 0 {
		vc.stats.AvgConversionTime = duration
	} else {
		vc.stats.AvgConversionTime = (vc.stats.AvgConversionTime*9 + duration) / 10
	}
}

func (vc *VideoConverter) recordFailure() {
	vc.mu.Lock()
	defer vc.mu.Unlock()
	vc.stats.TotalConversions++
	vc.stats.FailedConversions++
}

func (vc *VideoConverter) GetStats() VideoConverterStats {
	vc.mu.RLock()
	defer vc.mu.RUnlock()
	return vc.stats
}
