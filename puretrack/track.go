package puretrack

type Detection interface {
	GetXYXY() [4]float64
	GetScore() float64
}

type DetectionWithEmbedding interface {
	Detection
	GetEmbedding() []float64
}

type TrackState uint8

const (
	Tracked TrackState = iota
	Lost
)

type Track[T Detection] interface {
	GetDetection() T
	GetID() uint
	GetXYXY() [4]float64
}

type track[T Detection] struct {
	TrackState
	id                     uint
	activated              bool
	detection              T
	mean                   [8]float64
	cov                    [64]float64
	lastDetectionFrameIdx  uint
	firstDetectionFrameIdx uint
}

func (t *track[T]) GetDetection() T {
	return t.detection
}

func (t *track[T]) GetID() uint {
	return t.id
}

func (t *track[T]) GetXYXY() [4]float64 {
	var xyxy [4]float64
	copy(xyxy[:], t.mean[:4])
	w := xyxy[2]
	h := xyxy[3]
	xyxy[0] -= w / 2
	xyxy[1] -= h / 2
	xyxy[2] = xyxy[0] + w
	xyxy[3] = xyxy[1] + h
	return xyxy
}

func CmpTrackID[T Detection](t1, t2 Track[T]) int {
	if t1.GetID() < t2.GetID() {
		return -1
	}
	if t1.GetID() > t2.GetID() {
		return 1
	}
	return 0
}
