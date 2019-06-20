
import endo, time, utils, numpy
from os.path import abspath,join


# reload(endo)
# reload(utils)

path = join(abspath(''), 'Test_data', 'pat1_MRI_Layer_6.vtk')

polydata = utils.polydataReader(path)

landmarks = []

landmarks = utils.landmarkSelector(polydata, 2, [])

start = time.time(); MRI1 = endo.PyQCM(path, landmarks[0], landmarks[1]); print(time.time() - start)

homeomorphism = MRI1.homeomorphism

boundaryPoints  = homeomorphism[:,MRI1.boundary]
source          = numpy.zeros((boundaryPoints.shape[0],
                         boundaryPoints.shape[1] + 1))
destination     = numpy.zeros((boundaryPoints.shape[0],
                         boundaryPoints.shape[1] + 1))

source[:, 0:source.shape[1] - 1]        = boundaryPoints
source[:, source.shape[1] - 1]          = homeomorphism[:, MRI1.apex]

destination[:, 0:source.shape[1] - 1]   = boundaryPoints
destination[:, 0:source.shape[1] - 1]   = boundaryPoints

x = source[0,:]
y = source[1,:]
d = destination[0,:] + 1j*destination[1,:]

# thinPlateInterpolation = Rbf(x,y,d,function="thin_plate")
# result = thinPlateInterpolation(MRI1.homeomorphism[0,:], MRI1.homeomorphism[1,:])

thinPlateInterpolation = Rbf(source,numpy.zeros(source.shape),destination,function="thin_plate")
result                  = numpy.zeros(homeomorphism.shape) 
# result = thinPlateInterpolation(homeomorphism,numpy.zeros(source.shape))

# thinPlateInterpolation = RBFThinPlateSpline(x,y,d)
# start = time.time(); result = thinPlateInterpolation(homeomorphism[0,:], homeomorphism[0,:]); print(time.time() - start)

# homeomorphism[0,:] = result.real
# homeomorphism[1,:] = result.imag



for j in range(2, 5):
    start = time.time()
    k = 10**j
    for i in range(0, homeomorphism.shape[1], k):
        if (i + k) >= homeomorphism.shape[1]:
            result[:,i:] = thinPlateInterpolation(homeomorphism[:,i:])
        else:
            result[:,i:(i+k)] = thinPlateInterpolation(homeomorphism[:,i:(i+k)])
    print("time for k = " + str(k) + ": " + str(time.time() - start))


def tps(homeomorphism,k):
    result = numpy.zeros(homeomorphism.shape)
    for i in range(0, homeomorphism.shape[1]):
        if (i + k) >= homeomorphism.shape[1]:
            result[:,i:] = thinPlateInterpolation(homeomorphism[:,i:])
        else:
            result[:,i:(i+k)] = thinPlateInterpolation(homeomorphism[:,i:(i+k)])
    return result



with closing(Pool(processes=2)) as pool:
    for j in range(1, 51, 5):
        start = time.time()
        k = j
        result = pool.map(tps, [[k]])
    pool.terminate()



for j in range(1, 51, 5):
    start = time.time()
    k = j
    result = tps(homeomorphism, k)
    print("time for k = " + str(k) + ": " + str(time.time() - start))




start = time.time()
k = 1
for i in range(0, homeomorphism.shape[1], k):
    if (i + k) >= homeomorphism.shape[1]:
        result[:,i:] = thinPlateInterpolation(homeomorphism[:,i:])
    else:
        result[:,i:(i+k)] = thinPlateInterpolation(homeomorphism[:,i:(i+k)])


print("time for k = " + str(k) + ": " + str(time.time() - start))






for j in range(1, 51, 5):
    start = time.time()
    k = j
    # result = tps(homeomorphism, k)
    p = Process(target=tps, args=(homeomorphism,k))
    p.start()
    p.join()    
    print("time for k = " + str(k) + ": " + str(time.time() - start))






thinPlateInterpolation = RBFThinPlateSpline(x,y,d)

k=1000

for i in range(0, homeomorphism.shape[1], k):
    if (i + k) >= homeomorphism.shape[1]:
        # result[:,i:]        = thinPlateInterpolation(self.homeomorphism[:,i:])
        result              = thinPlateInterpolation(homeomorphism[0,i:],homeomorphism[1,i:])
    else:
        # result[:,i:(i+k)]   = thinPlateInterpolation(self.homeomorphism[:,i:(i+k)])
        result              = thinPlateInterpolation(homeomorphism[0,i:(i+k)],homeomorphism[1,i:(i+k)])


septo = 201479 - 1
apex  = 37963 - 1
