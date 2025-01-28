package lap

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLap(t *testing.T) {
	cost := [][]float64{
		{0.78, 0.2},
		{0.9, 1.0},
	}
	var unmatchedX, unmatchedY []int
	var matched [][2]int

	matched, unmatchedX, unmatchedY = SolveLinearAssignmentProblem(2, 2, cost, 1)
	require.Equal(t, matched, [][2]int{
		{0, 1},
		{1, 0},
	})
	require.Nil(t, unmatchedX)
	require.Nil(t, unmatchedY)
	matched, unmatchedX, unmatchedY = SolveLinearAssignmentProblem(2, 2, cost, 0.95)
	require.Equal(t, matched, [][2]int{
		{0, 1},
		{1, 0},
	})
	require.Nil(t, unmatchedX)
	require.Nil(t, unmatchedY)
	matched, unmatchedX, unmatchedY = SolveLinearAssignmentProblem(2, 2, cost, 0.8)
	require.Equal(t, matched, [][2]int{
		{0, 1},
	})
	require.Equal(t, unmatchedX, []int{1})
	require.Equal(t, unmatchedY, []int{0})

	cost = [][]float64{
		{0.78, 0.2},
		{0.9, 1.0},
		{1.0, 1.0},
		{0.1, 0.65},
	}

	matched, unmatchedX, unmatchedY = SolveLinearAssignmentProblem(4, 2, cost, 0.8)
	require.Equal(t, matched, [][2]int{
		{0, 1},
		{3, 0},
	})
	require.Equal(t, unmatchedX, []int{1, 2})
	require.Nil(t, unmatchedY, nil)
}
