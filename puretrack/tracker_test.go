package puretrack

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

type testDet struct {
	XYXY  [4]float64
	score float64
}

func (d testDet) GetXYXY() [4]float64 {
	return d.XYXY
}

func (d testDet) GetScore() float64 {
	return d.score
}

func TestTracker(t *testing.T) {
	tr := New[*testDet](BaseConfig)
	inputs, outputs := getTestData()

	for i := range inputs {
		dets := make([]*testDet, len(inputs[i]))
		for j := range inputs[i] {
			dets[j] = &testDet{
				XYXY:  [4]float64(inputs[i][j][:4]),
				score: inputs[i][j][4],
			}
		}
		tracked, _ := tr.Update(dets)

		got := make([][5]float64, len(tracked))
		for j := range tracked {
			xyxy := tracked[j].GetXYXY()
			for l := range xyxy {
				xyxy[l] = math.Round(xyxy[l]*1e6) / 1e6
			}
			got[j] = [5]float64{xyxy[0], xyxy[1], xyxy[2], xyxy[3], float64(tracked[j].GetID())}
		}
		require.Equal(t, outputs[i], got, fmt.Sprintf("frame %d", i+1))
	}
}

func BenchmarkTracker(b *testing.B) {
	tr := New[testDet](BaseConfig)

	inputs, _ := getTestData()

	for i := range b.N {
		i = i % len(inputs)

		dets := make([]testDet, len(inputs[i]))
		for j := range inputs[i] {
			dets[j] = testDet{
				XYXY:  [4]float64(inputs[i][j][:4]),
				score: inputs[i][j][4],
			}
		}

		tr.Update(dets)
	}
}
