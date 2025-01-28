package utils

import (
	"slices"

	"gonum.org/v1/gonum/mat"
)

func EmbeddingDistance(tracksEmbs, detsEmbs [][]float64) [][]float64 {
	if len(tracksEmbs) == 0 || len(detsEmbs) == 0 {
		return nil
	}

	rows := len(tracksEmbs)
	cols := len(tracksEmbs[0])

	costMatrixMat := mat.NewDense(rows, rows, nil)

	tracksEmbsMat := mat.NewDense(rows, cols, nil)
	for index, row := range tracksEmbs {
		tracksEmbsMat.SetRow(index, row)
	}

	detsEmbsMat := mat.NewDense(rows, cols, nil)
	for index, row := range detsEmbs {
		detsEmbsMat.SetRow(index, row)
	}

	for i := 0; i < rows; i++ {
		trackVec := tracksEmbsMat.RowView(i)

		for j := 0; j < rows; j++ {
			detVec := detsEmbsMat.RowView(j)
			sim := 0.
			for k := 0; k < cols; k++ {
				sim += trackVec.AtVec(k) * detVec.AtVec(k)
			}
			costMatrixMat.Set(i, j, 1-sim)
		}
	}

	costMatrix := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		costMatrix[i] = costMatrixMat.RawRowView(i)
	}

	return costMatrix
}

func FuseScore(costs [][]float64, confs []float64) [][]float64 {
	if len(costs) == 0 || len(confs) == 0 {
		return nil
	}

	rows := len(costs)
	cols := len(costs[0])

	confsExpanded := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			confsExpanded.Set(i, j, confs[j])
		}
	}

	costMatrix := mat.NewDense(rows, cols, nil)
	for index, row := range costs {
		costMatrix.SetRow(index, row)
	}
	iouSim := mat.NewDense(rows, cols, nil)
	iouSim.Apply(func(i, j int, v float64) float64 {
		return 1 - costMatrix.At(i, j)
	}, costMatrix)

	fuseSim := mat.NewDense(rows, cols, nil)
	fuseSim.MulElem(iouSim, confsExpanded)

	fuseCostMatrix := mat.NewDense(rows, cols, nil)
	fuseCostMatrix.Apply(func(i, j int, v float64) float64 {
		return 1 - fuseSim.At(i, j)
	}, fuseSim)

	fuseCosts := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		fuseCosts[i] = fuseCostMatrix.RawRowView(i)
	}

	return fuseCosts
}

func IoUDistBatch(bboxes1, bboxes2 [][4]float64) [][]float64 {
	rows1 := len(bboxes1)
	rows2 := len(bboxes2)

	if rows1 == 0 || rows2 == 0 {
		return nil
	}

	iou := make([][]float64, rows1)
	for i := 0; i < rows1; i++ {
		iou[i] = make([]float64, rows2)
	}

	for i := 0; i < rows1; i++ {
		for j := 0; j < rows2; j++ {
			b1X1, b1Y1, b1X2, b1Y2 := bboxes1[i][0], bboxes1[i][1], bboxes1[i][2], bboxes1[i][3]
			b2X1, b2Y1, b2X2, b2Y2 := bboxes2[j][0], bboxes2[j][1], bboxes2[j][2], bboxes2[j][3]

			xx1 := max(b1X1, b2X1)
			yy1 := max(b1Y1, b2Y1)
			xx2 := min(b1X2, b2X2)
			yy2 := min(b1Y2, b2Y2)

			interW := max(0.0, xx2-xx1)
			interH := max(0.0, yy2-yy1)
			interArea := interW * interH

			b1Area := (b1X2 - b1X1) * (b1Y2 - b1Y1)
			b2Area := (b2X2 - b2X1) * (b2Y2 - b2Y1)

			unionArea := b1Area + b2Area - interArea
			iou[i][j] = 1 - interArea/unionArea
		}
	}

	return iou
}

func SearchInMap[K comparable, V any](pool map[K]V, f func(V) bool) []K {
	var keys []K
	for k, v := range pool {
		if f(v) {
			keys = append(keys, k)
		}
	}
	return keys
}

func AdjustSliceSize[T any](slice []T, size int) []T {
	return slices.Grow(slice, size)[:size]
}
