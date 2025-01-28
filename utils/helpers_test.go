package utils

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEmbeddingDistance(t *testing.T) {
	tracksEmbs := [][]float64{
		{0.26726124, 0.53452248, 0.80178373},
		{0.45584231, 0.56980288, 0.68376346},
	}
	detsEmbs := [][]float64{
		{0.11624764, 0.34874292, 0.92998111},
		{0.42426407, 0.56568542, 0.707106786},
	}
	expected := [][]float64{
		{0.03687686, 0.01729237},
		{0.11240759, 0.00077952},
	}

	out := EmbeddingDistance(tracksEmbs, detsEmbs)
	require.InDeltaSlice(t, expected[0], out[0], 1e-8)
	require.InDeltaSlice(t, expected[1], out[1], 1e-8)

	require.Nil(t, EmbeddingDistance(nil, detsEmbs))
	require.Nil(t, EmbeddingDistance(tracksEmbs, nil))
}

func TestFuseScore(t *testing.T) {
	costMatrix := [][]float64{
		{0.78, 0.2},
		{0.9, 1.0},
	}
	confs := []float64{0.8, 0.9}
	expected := [][]float64{
		{0.824, 0.28},
		{0.92, 1.0},
	}

	out := FuseScore(costMatrix, confs)
	require.InDeltaSlice(t, expected[0], out[0], 1e-8)
	require.InDeltaSlice(t, expected[1], out[1], 1e-8)

	require.Nil(t, FuseScore(nil, confs))
	require.Nil(t, FuseScore(costMatrix, nil))
}

// func TestIoUBatch(t *testing.T) {
// 	boxes1 := [][4]float64{
// 		{50, 50, 100, 100},
// 		{30, 30, 50, 50},
// 		{200, 200, 150, 150},
// 		{400, 400, 100, 100},
// 	}
// 	boxes2 := [][4]float64{
// 		{60, 60, 80, 80},
// 		{0, 0, 100, 100},
// 		{450, 450, 50, 50},
// 	}
// 	expected := [][]float64{
// 		{.16, .25, 0.},
// 		{0., .04, 0.},
// 		{0., 0., 0.},
// 		{0., 0., 0.},
// 	}

// 	out := IoUDistBatch(boxes1, boxes2)
// 	require.InDeltaSlice(t, expected[0], out[0], 1e-8)
// 	require.InDeltaSlice(t, expected[1], out[1], 1e-8)
// 	require.InDeltaSlice(t, expected[2], out[2], 1e-8)
// 	require.InDeltaSlice(t, expected[3], out[3], 1e-8)

// 	require.Nil(t, IoUDistBatch(nil, boxes2))
// 	require.Nil(t, IoUDistBatch(boxes1, nil))
// }
