package lap

import (
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

//#include <stdlib.h>
//#include "lapjv.cpp"
import "C"

func SolveLinearAssignmentProblem(rows, cols int, costs [][]float64, limit float64) ([][2]int, []int, []int) {
	if rows == 0 {
		var unmatchedY []int
		for i := 0; i < cols; i++ {
			unmatchedY = append(unmatchedY, i)
		}
		return nil, nil, unmatchedY
	}
	if cols == 0 {
		var unmatchedX []int
		for i := 0; i < rows; i++ {
			unmatchedX = append(unmatchedX, i)
		}
		return nil, unmatchedX, nil
	}
	n := rows

	costMatrix := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		costMatrix.SetRow(i, costs[i])
	}

	n = rows + cols
	cMatrix := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i < rows && j < cols {
				cMatrix.Set(i, j, costMatrix.At(i, j))
			} else if i >= rows && j >= cols {
				cMatrix.Set(i, j, 0)
			} else {
				cMatrix.Set(i, j, limit/2)
			}
		}
	}

	var rowPtr *C.double
	doubleSize := (C.int)(int(unsafe.Sizeof(rowPtr)))
	cCostPtr := (**C.double)(C.malloc(C.ulong(C.int(n) * doubleSize)))

	costPtr := unsafe.Slice(cCostPtr, n)
	for i := 0; i < n; i++ {
		costPtr[i] = (*C.double)(unsafe.Pointer(&cMatrix.RawRowView(i)[0]))
	}
	xIdxs := make([]int32, n)
	yIdxs := make([]int32, n)
	cX := (*C.int)(unsafe.Pointer(&xIdxs[0]))
	cY := (*C.int)(unsafe.Pointer(&yIdxs[0]))

	C.lapjv_internal(C.uint(n), cCostPtr, cX, cY)

	for i, x := range xIdxs {
		if int(x) >= cols {
			xIdxs[i] = -1
		}
	}

	for i, y := range yIdxs {
		if int(y) >= rows {
			yIdxs[i] = -1
		}
	}

	xIdxs = xIdxs[:rows]
	yIdxs = yIdxs[:cols]

	var matches [][2]int
	var unmatchedX, unmatchedY []int

	for i, x := range xIdxs {
		if x >= 0 {
			matches = append(matches, [2]int{i, int(x)})
		} else {
			unmatchedX = append(unmatchedX, i)
		}
	}

	for i, y := range yIdxs {
		if y < 0 {
			unmatchedY = append(unmatchedY, i)
		}
	}

	return matches, unmatchedX, unmatchedY
}
