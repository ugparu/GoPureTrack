package puretrack

import (
	"slices"

	"github.com/ugparu/GoPureTrack/kalman"
	"github.com/ugparu/GoPureTrack/utils"
	"github.com/ugparu/GoPureTrack/utils/lap"
)

type Config struct {
	trackNewThresh  float64
	trackHighThresh float64
	trackLowThresh  float64
	matchThresh     float64
	maxTimeLost     uint
}

const initPoolSize = 2 << 5

var BaseConfig = Config{
	trackNewThresh:  0.6,
	trackHighThresh: 0.6,
	trackLowThresh:  0.1,
	matchThresh:     0.9,
	maxTimeLost:     30,
}

type Tracker[T Detection] struct {
	Config
	tracksPool map[uint]*track[T]
	trackID    uint
	frameID    uint
}

func New[T Detection](config Config) *Tracker[T] {
	return &Tracker[T]{
		Config:     config,
		tracksPool: make(map[uint]*track[T], initPoolSize),
		trackID:    1,
		frameID:    1,
	}
}

func (t *Tracker[T]) Update(detsXYXY []T) ([]Track[T], []Track[T]) {
	// Split incoming detections into high and low confidence detections
	detsHigh, detsLow := t.splitDets(detsXYXY)

	// Tracked track ids with only one detection
	unconfirmedTracksIndexes := utils.SearchInMap(t.tracksPool, func(track *track[T]) bool {
		return track.TrackState == Tracked && !track.activated
	})

	// Tracked track ids with >=2 detections or lost tracks
	confirmedTracksIndexes := utils.SearchInMap(t.tracksPool, func(track *track[T]) bool {
		return track.activated || track.TrackState == Lost
	})

	means := make([][8]float64, len(confirmedTracksIndexes))
	covs := make([][64]float64, len(confirmedTracksIndexes))
	tracksBoxes := make([][4]float64, len(confirmedTracksIndexes))
	for i, id := range confirmedTracksIndexes {
		means[i] = t.tracksPool[id].mean
		if t.tracksPool[id].TrackState != Tracked {
			means[i][7] = 0
		}
		covs[i] = t.tracksPool[id].cov
	}

	// Predicting for all tracks
	kalman.GetFilter().MultiPredict(means, covs)

	for i, id := range confirmedTracksIndexes {
		t.tracksPool[id].mean = means[i]
		t.tracksPool[id].cov = covs[i]

		tracksBoxes[i] = t.tracksPool[id].GetXYXY()
	}

	// Calculating distances for high confidence detections and confirmed tracks
	detsBoxes := make([][4]float64, len(detsHigh))
	detsConfs := make([]float64, len(detsHigh))
	for i, det := range detsHigh {
		detsBoxes[i] = det.GetXYXY()
		detsConfs[i] = det.GetScore()
	}

	var dists [][]float64
	if len(detsHigh) != 0 {
		dists = utils.IoUDistBatch(tracksBoxes, detsBoxes)
		dists = utils.FuseScore(dists, detsConfs)

		castedDetections := make([]Detection, len(detsHigh))
		for i, det := range detsHigh {
			castedDetections[i] = det
		}

		if _, ok := castedDetections[0].(DetectionWithEmbedding); ok {
			trackEmbeddings := make([][]float64, len(confirmedTracksIndexes))
			for i, id := range confirmedTracksIndexes {
				var d Detection
				_ = d
				d = t.tracksPool[id].detection
				trackEmbeddings[i] = d.(DetectionWithEmbedding).GetEmbedding()
			}
			detEmbeddings := make([][]float64, len(detsHigh))
			for i, det := range castedDetections {
				detEmbeddings[i] = det.(DetectionWithEmbedding).GetEmbedding()
			}
			embDists := utils.EmbeddingDistance(trackEmbeddings, detEmbeddings)
			for i, row := range embDists {
				for j, embDist := range row {
					if embDist < dists[i][j] {
						dists[i][j] = embDist
					}
				}
			}
		}
	}

	// Solving lap for high confidence detections and confirmed tracks
	assignments, unmatchedTracks, unmatchedDets := lap.SolveLinearAssignmentProblem(len(confirmedTracksIndexes), len(detsHigh), dists, t.matchThresh)

	// Updating matched confirmed tracks and high detections
	means = means[:len(assignments)]
	covs = covs[:len(assignments)]
	measurments := make([][4]float64, len(assignments))
	for i, match := range assignments {
		trackID := confirmedTracksIndexes[match[0]]
		det := detsHigh[match[1]]

		means[i] = t.tracksPool[trackID].mean
		covs[i] = t.tracksPool[trackID].cov

		measurments[i] = det.GetXYXY()
		measurments[i][2] -= measurments[i][0]
		measurments[i][3] -= measurments[i][1]
		measurments[i][0] += measurments[i][2] / 2
		measurments[i][1] += measurments[i][3] / 2
	}

	kalman.GetFilter().MultiUpdate(means, covs, measurments)

	for i, match := range assignments {
		trackID := confirmedTracksIndexes[match[0]]
		det := detsHigh[match[1]]

		t.tracksPool[trackID].mean = means[i]
		t.tracksPool[trackID].cov = covs[i]
		t.tracksPool[trackID].detection = det
		t.tracksPool[trackID].TrackState = Tracked
		t.tracksPool[trackID].activated = true
		t.tracksPool[trackID].lastDetectionFrameIdx = t.frameID
	}

	var unmatchedConfirmedTracksIndexes []uint
	for _, index := range unmatchedTracks {
		if t.tracksPool[confirmedTracksIndexes[index]].TrackState == Tracked {
			unmatchedConfirmedTracksIndexes = append(unmatchedConfirmedTracksIndexes, confirmedTracksIndexes[index])
		}
	}

	unmatchedDetsHigh := make([]T, len(unmatchedDets))
	for i, index := range unmatchedDets {
		unmatchedDetsHigh[i] = detsHigh[index]
	}

	tracksBoxes = slices.Grow(tracksBoxes, len(unmatchedConfirmedTracksIndexes))[:len(unmatchedConfirmedTracksIndexes)]
	for i, id := range unmatchedConfirmedTracksIndexes {
		tracksBoxes[i] = t.tracksPool[id].GetXYXY()
	}

	detsBoxes = slices.Grow(detsBoxes, len(detsLow))[:len(detsLow)]
	for i, det := range detsLow {
		detsBoxes[i] = det.GetXYXY()
	}

	dists = utils.IoUDistBatch(tracksBoxes, detsBoxes)

	assignments, unmatchedTracks, _ = lap.SolveLinearAssignmentProblem(len(unmatchedConfirmedTracksIndexes), len(detsLow), dists, 0.5)
	// fmt.Printf("%v\n", unmatchedConfirmedTracksIndexes)

	means = slices.Grow(means, len(assignments))[:len(assignments)]
	covs = slices.Grow(covs, len(assignments))[:len(assignments)]
	measurments = slices.Grow(measurments, len(assignments))[:len(assignments)]
	for i, match := range assignments {
		trackID := unmatchedConfirmedTracksIndexes[match[0]]
		det := detsLow[match[1]]

		means[i] = t.tracksPool[trackID].mean
		covs[i] = t.tracksPool[trackID].cov
		measurments[i] = det.GetXYXY()
		measurments[i][2] -= measurments[i][0]
		measurments[i][3] -= measurments[i][1]
		measurments[i][0] += measurments[i][2] / 2
		measurments[i][1] += measurments[i][3] / 2
	}

	kalman.GetFilter().MultiUpdate(means, covs, measurments)

	for i, match := range assignments {
		trackID := unmatchedConfirmedTracksIndexes[match[0]]
		det := detsLow[match[1]]

		t.tracksPool[trackID].mean = means[i]
		t.tracksPool[trackID].cov = covs[i]
		t.tracksPool[trackID].detection = det
		t.tracksPool[trackID].TrackState = Tracked
		t.tracksPool[trackID].activated = true
		t.tracksPool[trackID].lastDetectionFrameIdx = t.frameID
	}

	for _, index := range unmatchedTracks {
		t.tracksPool[unmatchedConfirmedTracksIndexes[index]].TrackState = Lost
	}

	tracksBoxes = make([][4]float64, len(unconfirmedTracksIndexes))
	for i, id := range unconfirmedTracksIndexes {
		tracksBoxes[i] = t.tracksPool[id].GetXYXY()
	}

	detsBoxes = make([][4]float64, len(unmatchedDetsHigh))
	for i, det := range unmatchedDetsHigh {
		detsBoxes[i] = det.GetXYXY()
	}

	dists = utils.IoUDistBatch(tracksBoxes, detsBoxes)

	assignments, unmatchedTracks, unmatchedDets = lap.SolveLinearAssignmentProblem(len(unconfirmedTracksIndexes), len(unmatchedDetsHigh), dists, 0.7)

	means = slices.Grow(means, len(assignments))[:len(assignments)]
	covs = slices.Grow(covs, len(assignments))[:len(assignments)]
	measurments = slices.Grow(measurments, len(assignments))[:len(assignments)]
	for i, match := range assignments {
		trackID := unconfirmedTracksIndexes[match[0]]
		det := unmatchedDetsHigh[match[1]]

		means[i] = t.tracksPool[trackID].mean
		covs[i] = t.tracksPool[trackID].cov
		measurments[i] = det.GetXYXY()
		measurments[i][2] -= measurments[i][0]
		measurments[i][3] -= measurments[i][1]
		measurments[i][0] += measurments[i][2] / 2
		measurments[i][1] += measurments[i][3] / 2
	}

	kalman.GetFilter().MultiUpdate(means, covs, measurments)

	for i, match := range assignments {
		trackID := unconfirmedTracksIndexes[match[0]]
		det := unmatchedDetsHigh[match[1]]

		t.tracksPool[trackID].mean = means[i]
		t.tracksPool[trackID].cov = covs[i]
		t.tracksPool[trackID].detection = det
		t.tracksPool[trackID].TrackState = Tracked
		t.tracksPool[trackID].activated = true
		t.tracksPool[trackID].lastDetectionFrameIdx = t.frameID
	}

	for _, index := range unmatchedTracks {
		delete(t.tracksPool, unconfirmedTracksIndexes[index])
	}

	detsToTracksBoxes := make([][4]float64, len(unmatchedDets))
	detsToTracks := make([]T, len(unmatchedDets))
	for i, index := range unmatchedDets {
		detsToTracks[i] = unmatchedDetsHigh[index]
		detsToTracksBoxes[i] = unmatchedDetsHigh[index].GetXYXY()
		detsToTracksBoxes[i][2] -= detsToTracksBoxes[i][0]
		detsToTracksBoxes[i][3] -= detsToTracksBoxes[i][1]
		detsToTracksBoxes[i][0] += detsToTracksBoxes[i][2] / 2
		detsToTracksBoxes[i][1] += detsToTracksBoxes[i][3] / 2
	}

	means, covs = kalman.GetFilter().MultiInitiate(detsToTracksBoxes)

	for i := range means {
		t.tracksPool[t.trackID] = &track[T]{
			TrackState:             Tracked,
			id:                     t.trackID,
			activated:              t.frameID == 1,
			detection:              detsToTracks[i],
			mean:                   means[i],
			cov:                    covs[i],
			lastDetectionFrameIdx:  t.frameID,
			firstDetectionFrameIdx: t.frameID,
		}
		t.trackID++
	}

	var removedTracks []Track[T]
	var trackedTracks []Track[T]
	for id, track := range t.tracksPool {
		if track.TrackState == Lost && t.frameID-track.lastDetectionFrameIdx > t.maxTimeLost {
			removedTracks = append(removedTracks, track)
			delete(t.tracksPool, id)
		}
	}

	var trackedBoxes, lostBoxes [][4]float64
	var trackedIdxs, lostIdxs []uint

	for _, track := range t.tracksPool {
		if track.TrackState == Tracked {
			trackedBoxes = append(trackedBoxes, track.GetXYXY())
			trackedIdxs = append(trackedIdxs, track.id)
		} else {
			lostBoxes = append(lostBoxes, track.GetXYXY())
			lostIdxs = append(lostIdxs, track.id)
		}
	}

	dists = utils.IoUDistBatch(trackedBoxes, lostBoxes)

	for ti, row := range dists {
		for li, x := range row {
			if x < 0.15 {
				id1 := trackedIdxs[ti]
				id2 := lostIdxs[li]
				t1, found := t.tracksPool[id1]
				if !found {
					continue
				}
				t2, found := t.tracksPool[id2]
				if !found {
					continue
				}
				dur1 := t1.lastDetectionFrameIdx - t1.firstDetectionFrameIdx
				dur2 := t2.lastDetectionFrameIdx - t2.firstDetectionFrameIdx

				if dur1 > dur2 {
					delete(t.tracksPool, id2)
				} else if dur2 > dur1 {
					delete(t.tracksPool, id1)
				}
			}
		}
	}

	for _, track := range t.tracksPool {
		if track.TrackState == Tracked && track.activated {
			trackedTracks = append(trackedTracks, track)
		}
	}

	slices.SortFunc(trackedTracks, CmpTrackID[T])
	slices.SortFunc(removedTracks, CmpTrackID[T])

	t.frameID++

	return trackedTracks, removedTracks
}

func (t *Tracker[T]) splitDets(detsXYXY []T) (detsHigh, detsLow []T) {
	rows := len(detsXYXY)

	indsHigh := make([]int, 0)
	indsLow := make([]int, 0)
	for i := 0; i < rows; i++ {
		conf := detsXYXY[i].GetScore()
		if conf > t.trackHighThresh {
			indsHigh = append(indsHigh, i)
		} else if conf > t.trackLowThresh && conf <= t.trackHighThresh {
			indsLow = append(indsLow, i)
		}
	}

	detsHigh = make([]T, len(indsHigh))
	detsLow = make([]T, len(indsLow))

	for i, ind := range indsHigh {
		detsHigh[i] = detsXYXY[ind]
	}

	for i, ind := range indsLow {
		detsLow[i] = detsXYXY[ind]
	}

	return detsHigh, detsLow
}
