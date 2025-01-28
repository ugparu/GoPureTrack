package kalman

import (
	"sync"
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

const (
	stdWeightPosition = 0.05
	stdWeightVelocity = 0.00625

	stdWeightPositionx2  = stdWeightPosition * 2
	stdWeightVelocityx10 = stdWeightVelocity * 10

	ndim = 4
)

var (
	stdPosVec = mat.NewVecDense(4, []float64{stdWeightPosition, stdWeightPosition, stdWeightPosition, stdWeightPosition})
	stdVelVec = mat.NewVecDense(4, []float64{stdWeightVelocity, stdWeightVelocity, stdWeightVelocity, stdWeightVelocity})

	stdPosx2Velx10Vec = mat.NewVecDense(8, []float64{stdWeightPositionx2, stdWeightPositionx2, stdWeightPositionx2, stdWeightPositionx2, stdWeightVelocityx10, stdWeightVelocityx10, stdWeightVelocityx10, stdWeightVelocityx10})
)

type Filter interface {
	MultiPredict(means [][8]float64, covs [][64]float64)
	MultiUpdate(means [][8]float64, covs [][64]float64, measurements [][4]float64)
	MultiInitiate(m [][4]float64) ([][8]float64, [][64]float64)
}

type filter struct {
	MotionMat *mat.Dense
	UpdateMat *mat.Dense
}

var once = new(sync.Once)
var filterInstance *filter

func GetFilter() Filter {
	once.Do(func() {
		filterInstance = newBaseKalmanFilter()
	})
	return filterInstance
}

func newBaseKalmanFilter() *filter {
	dt := 1.0
	motionMat := mat.NewDense(2*ndim, 2*ndim, nil)
	updateMat := mat.NewDense(ndim, 2*ndim, nil)

	for i := 0; i < 2*ndim; i++ {
		motionMat.Set(i, i, 1.0)
		if i < ndim {
			motionMat.Set(i, ndim+i, dt)
		}
	}

	for i := 0; i < ndim; i++ {
		updateMat.Set(i, i, 1.0)
	}

	return &filter{
		MotionMat: motionMat,
		UpdateMat: updateMat,
	}
}

func (kf *filter) GetInitialCovariancesStd(measurements [][4]float64) [][8]float64 {
	result := make([][8]float64, len(measurements))
	for i, m := range measurements {
		m2 := m[2]
		m3 := m[3]

		m23232323 := mat.NewVecDense(8, []float64{m2, m3, m2, m3, m2, m3, m2, m3})
		m23232323.MulElemVec(stdPosx2Velx10Vec, m23232323)
		result[i] = [8]float64(m23232323.RawVector().Data)
	}

	return result
}

func (kf *filter) GetMeasurementNoisesStd(means [][4]float64) [][4]float64 {
	noisesStd := make([][4]float64, len(means))
	for i, m := range means {
		m2 := m[2]
		m3 := m[3]

		m2323 := mat.NewVecDense(4, []float64{m2, m3, m2, m3})
		m2323.MulElemVec(m2323, stdPosVec)
		noisesStd[i] = [4]float64(m2323.RawVector().Data)
	}

	return noisesStd
}

func (kf *filter) GetMultiProcessNoiseStd(means [][4]float64) ([][4]float64, [][4]float64) {
	stdPos := make([][4]float64, len(means))
	stdVel := make([][4]float64, len(means))

	for i, m := range means {
		m2 := m[2]
		m3 := m[3]

		m2323 := mat.NewVecDense(4, []float64{m2, m3, m2, m3})
		m2323.MulElemVec(m2323, stdPosVec)
		stdPos[i] = [4]float64(m2323.RawVector().Data)

		m2323.SetVec(0, m2)
		m2323.SetVec(1, m3)
		m2323.SetVec(2, m2)
		m2323.SetVec(3, m3)
		m2323.MulElemVec(m2323, stdVelVec)
		stdVel[i] = [4]float64(m2323.RawVector().Data)
	}

	return stdPos, stdVel
}

func (df *filter) MultiInitiate(m [][4]float64) ([][8]float64, [][64]float64) {
	mean := make([][8]float64, len(m))
	covariances := make([][64]float64, len(m))

	for i, measurement := range m {
		copy(mean[i][:4], measurement[:])

		std := df.GetInitialCovariancesStd([][4]float64{measurement})[0]
		for j := 0; j < 8; j++ {
			covariances[i][j*8+j] = std[j] * std[j]
		}
	}

	return mean, covariances
}

func (df *filter) MultiPredict(means [][8]float64, covs [][64]float64) {
	if len(means) == 0 {
		return
	}

	croppedMeans := make([][4]float64, len(means))
	for i, m := range means {
		copy(croppedMeans[i][:], m[:4])
	}

	pos, vel := df.GetMultiProcessNoiseStd(croppedMeans)

	c := len(means)
	sqr := mat.NewDense(c, 8, nil)
	for i := 0; i < c; i++ {
		sqr.SetRow(i, append(pos[i][:], vel[i][:]...))
	}
	sqr.MulElem(sqr, sqr)

	motionCov := make([]*mat.Dense, c)
	for i := 0; i < c; i++ {
		motionCov[i] = mat.NewDense(8, 8, nil)
		for j := 0; j < 8; j++ {
			motionCov[i].Set(j, j, sqr.At(i, j))
		}
	}

	mean := mat.NewDense(c, 8, unsafe.Slice(&means[0][0], 8*c))
	mean.Mul(mean, df.MotionMat.T())

	left := mat.NewDense(8, 8, nil)
	for i := 0; i < c; i++ {
		covMat := mat.NewDense(8, 8, covs[i][:])
		left.Mul(df.MotionMat, covMat)
		covMat.Mul(left, df.MotionMat.T())
		covMat.Add(covMat, motionCov[i])
	}
}

func (df *filter) MultiProject(means [][8]float64, covs [][64]float64) ([][4]float64, [][16]float64) {
	croppedMeans := make([][4]float64, len(means))
	for i, m := range means {
		copy(croppedMeans[i][:], m[:4])
	}

	stds := df.GetMeasurementNoisesStd(croppedMeans)
	stdsMat := mat.NewDense(len(means), 4, unsafe.Slice(&stds[0][0], 4*len(means)))

	c := len(means)
	innovationCovs := make([]*mat.Dense, c)
	for i := 0; i < c; i++ {
		innovationCovs[i] = mat.NewDense(4, 4, nil)
		for j := 0; j < 4; j++ {
			innovationCovs[i].Set(j, j, stdsMat.At(i, j)*stdsMat.At(i, j))
		}
	}

	projectedMeans := make([][4]float64, c)
	projectedMeansMat := mat.NewDense(c, 4, unsafe.Slice(&projectedMeans[0][0], 4*c))
	mean := mat.NewDense(c, 8, unsafe.Slice(&means[0][0], 8*c))
	projectedMeansMat.Mul(mean, df.UpdateMat.T())

	projectedCovs := make([][16]float64, c)
	for i := 0; i < c; i++ {
		projectedCovMat := mat.NewDense(4, 4, unsafe.Slice(&projectedCovs[i][0], 4*4))
		covMat := mat.NewDense(8, 8, covs[i][:])

		tmpMat := mat.NewDense(4, 8, nil)
		tmpMat.Mul(df.UpdateMat, covMat)
		projectedCovMat.Mul(tmpMat, df.UpdateMat.T())
		projectedCovMat.Add(projectedCovMat, innovationCovs[i])
	}

	return projectedMeans, projectedCovs
}

func (df *filter) MultiUpdate(means [][8]float64, covs [][64]float64, measurements [][4]float64) {
	if len(means) == 0 {
		return
	}

	projectedMeans, projectedCovs := df.MultiProject(means, covs)

	projectedMeansMat := mat.NewDense(len(means), 4, unsafe.Slice(&projectedMeans[0][0], 4*len(means)))

	measurementsMat := mat.NewDense(len(means), 4, unsafe.Slice(&measurements[0][0], 4*len(measurements)))

	c := len(means)
	innovationMeans := mat.NewDense(c, 4, nil)
	innovationMeans.Sub(measurementsMat, projectedMeansMat)

	kalmanGain := mat.NewDense(4, 8, nil)
	innovations := mat.NewDense(c, 8, nil)

	kalmanCov := mat.NewDense(8, 4, nil)
	newCovariance := mat.NewDense(8, 8, nil)

	for i := 0; i < c; i++ {
		projectedCovMat := mat.NewDense(4, 4, unsafe.Slice(&projectedCovs[i][0], 4*4))
		covMat := mat.NewDense(8, 8, covs[i][:])

		ch := &mat.Cholesky{}
		ch.Factorize(projectedCovMat.DiagView())

		updatedCovariance := mat.NewDense(8, 4, nil)
		updatedCovariance.Mul(covMat, df.UpdateMat.T())

		ch.SolveTo(kalmanGain, updatedCovariance.T())

		innovations.Slice(i, i+1, 0, 8).(*mat.Dense).Mul(innovationMeans.Slice(i, i+1, 0, 4), kalmanGain)

		kalmanCov.Mul(kalmanGain.T(), projectedCovMat)

		newCovariance.Mul(kalmanCov, kalmanGain)
		covMat.Sub(covMat, newCovariance)
	}

	meansMat := mat.NewDense(c, 8, unsafe.Slice(&means[0][0], 8*c))
	meansMat.Add(meansMat, innovations)
}
