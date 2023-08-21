import numpy as np
import torch
def six_ellipsoid_parameters(majorAxis,minorAxis_1,minorAxis_2):
    
## takes ellipsoid axes in Cartesian coordinates and returns'
## six coefficients that describe the surface of the ellipsoid as
## (see https://math.stackexchange.com/questions/1865188/how-to-prove-the-parallel-projection-of-an-ellipsoid-is-an-ellipse)
##
##   A x^2 + B y^2 + C z^2 + 2(Dxy + Exz +Fyz) = 1
##
## note that this notation is NOT universal; the wikipedia article at
## https://en.wikipedia.org/wiki/Ellipse uses a similar, but different 
## in detail, notation.

#
  print("have entered six_ellipsoid_parameters")
  print("  ")
  print(" ")
## 
##  majorAxis, minorAxis_1, and minorAxis_2 are jagged arrrays --
##  each event has a variable number of tracks, and each track
##  has three entries corresponding to the lengths of the 
##  x, y, and z components of the axes.  
##  The "usual" numpy methods for manipulating these do not
##  always work as these *assume* fixed array structures
##  It *appears* the hacks used here suffice 

##  first for each track, for each axis, find the length squared
  mag_3_sq = np.multiply(majorAxis[:,0],majorAxis[:,0]) 
  mag_3_sq = mag_3_sq + np.multiply(majorAxis[:,1],majorAxis[:,1]) 
  mag_3_sq = mag_3_sq + np.multiply(majorAxis[:,2],majorAxis[:,2])

  mag_2_sq = np.multiply(minorAxis_2[:,0],minorAxis_2[:,0]) 
  mag_2_sq = mag_2_sq + np.multiply(minorAxis_2[:,1],minorAxis_2[:,1]) 
  mag_2_sq = mag_2_sq + np.multiply(minorAxis_2[:,2],minorAxis_2[:,2])

  mag_1_sq = np.multiply(minorAxis_1[:,0],minorAxis_1[:,0]) 
  mag_1_sq = mag_1_sq + np.multiply(minorAxis_1[:,1],minorAxis_1[:,1]) 
  mag_1_sq = mag_1_sq + np.multiply(minorAxis_1[:,2],minorAxis_1[:,2])

  nEvts = len(majorAxis)
  print("  nEvts = ",nEvts)

## by creating u1, u2, and u3 as copies of the axes,
## they acquire the correct array structure
  u1 = minorAxis_1
  u2 = minorAxis_2 
  u3 = majorAxis

##  this is an ugly, brute force hack, but it
##  seems to work
  for iEvt in range(nEvts):
    nTrks = len(u3[iEvt][0])
    if (iEvt < 10):
      print(" iEvt, nTrks = ", iEvt, nTrks)
    for iTrk in range(nTrks):
      u3[iEvt][0][iTrk] = u3[iEvt][0][iTrk]/mag_3_sq[iEvt][iTrk]
      u3[iEvt][1][iTrk] = u3[iEvt][1][iTrk]/mag_3_sq[iEvt][iTrk]
      u3[iEvt][2][iTrk] = u3[iEvt][2][iTrk]/mag_3_sq[iEvt][iTrk]

      u2[iEvt][0][iTrk] = u2[iEvt][0][iTrk]/mag_2_sq[iEvt][iTrk]
      u2[iEvt][1][iTrk] = u2[iEvt][1][iTrk]/mag_2_sq[iEvt][iTrk]
      u2[iEvt][2][iTrk] = u2[iEvt][2][iTrk]/mag_2_sq[iEvt][iTrk]

      u1[iEvt][0][iTrk] = u1[iEvt][0][iTrk]/mag_1_sq[iEvt][iTrk]
      u1[iEvt][1][iTrk] = u1[iEvt][1][iTrk]/mag_1_sq[iEvt][iTrk]
      u1[iEvt][2][iTrk] = u1[iEvt][2][iTrk]/mag_1_sq[iEvt][iTrk]

##  because u1, u2, and u3 have the original axis structures,
##  it seems we can use the standard numpy method for these
##  calculations
  A = u1[:,0]*u1[:,0] + u2[:,0]*u2[:,0] + u3[:,0]*u3[:,0]
  B = u1[:,1]*u1[:,1] + u2[:,1]*u2[:,1] + u3[:,1]*u3[:,1]
  C = u1[:,2]*u1[:,2] + u2[:,2]*u2[:,2] + u3[:,2]*u3[:,2]

  D = np.multiply(u1[:,0],u1[:,1]) + np.multiply(u2[:,0],u2[:,1]) + np.multiply(u3[:,0],u3[:,1])
  E = np.multiply(u1[:,2],u1[:,0]) + np.multiply(u2[:,2],u2[:,0]) + np.multiply(u3[:,2],u3[:,0])
  F = np.multiply(u1[:,1],u1[:,2]) + np.multiply(u2[:,1],u2[:,2]) + np.multiply(u3[:,1],u3[:,2])

## mds   D = u1[:,0]*u1[:,1] + u2[:,0]*u2[:,1] + u3[:,0]*u3[:,1]
## mds   E = u1[:,2]*u1[:,0] + u2[:,2]*u2[:,0] + u3[:,2]*u3[:,0]
## mds   F = u1[:,1]*u1[:,2] + u2[:,1]*u2[:,2] + u3[:,1]*u3[:,2]

## as a sanity check, let's print out some of the details inputs
## and outputs so we can check them by hand


  
  return A, B, C, D, E ,F


from contextlib import contextmanager, redirect_stdout, redirect_stderr
import sys
import time


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""

    __slots__ = ("file", "progress")

    def __init__(self, file, progress):
        self.file = file
        self.progress = progress

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            self.progress.write(x.strip(), file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


@contextmanager
def tqdm_redirect(progress):
    old_out = sys.stdout

    if hasattr(progress, "postfix"):
        with redirect_stdout(DummyTqdmFile(sys.stdout, progress)), redirect_stderr(
            DummyTqdmFile(sys.stderr, progress)
        ):
            yield old_out
    else:
        yield old_out


def import_progress_bar(notebook):
    """Set up notebook or regular progress bar.

    If None or if piping to a file, just provide an empty do-nothing function."""

    def progress(iterator, **kargs):
        return iterator

## mds 220731    if notebook is None:
## mds 220731        pass
## mds 220731    elif notebook:
## mds 220731        from tqdm import tqdm_notebook as progress
## mds 220731    elif sys.stdout.isatty():
## mds 220731        from tqdm import tqdm as progress
## mds 220731    else:
## mds 220731        # Don't display progress if this is not a
## mds 220731        # notebook and not connected to the terminal
## mds 220731        pass
## mds 220731
##
    pass
## mds 220731

    return progress


class Timer(object):
    __slots__ = "message verbose start_time".split()

    def __init__(self, message=None, start=None, verbose=True):
        """
        If message is None, add a default message.
        If start is not None, then print start then message.
        Turn off all printing with verbose.
        """

        if verbose and start is not None:
            print(start, end="", flush=True)
        if message is not None:
            self.message = message
        elif start is not None:
            self.message = " took {time:.4} s"
        else:
            self.message = "Operation took {time:.4} s"

        self.verbose = verbose
        self.start_time = time.time()

    def elapsed_time(self):
        return time.time() - self.start_time

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if self.verbose:
            print(self.message.format(time=self.elapsed_time()))


def get_device_from_model(model):
    if hasattr(model, "weight"):
        return model.weight.device
    else:
        return get_device_from_model(list(model.children())[0])

try:
    import awkward0 as awkward
except ModuleNotFoundError:
    import awkward
import numpy as np


def concatenate(jaggedarrays):
    """
    Concatenate jagged arrays. Does not support alternate `axis` or `out=`. Requires 1 or more jaggedarrays.
    """

    # Support generators:
    jaggedarrays = list(jaggedarrays)

    # Propogate Awkward 0.8+ jagged array types
    first = jaggedarrays[0]
    JaggedArray = getattr(first, "JaggedArray", awkward.JaggedArray)

    # Perform the concatenation
    contents = np.concatenate([j.flatten() for j in jaggedarrays])
    counts = np.concatenate([j.counts for j in jaggedarrays])
    return JaggedArray.fromcounts(counts, contents)


from torch.utils.data import TensorDataset

from functools import partial
import warnings
from collections import namedtuple


## add 220817 for scrubbing code
import math

# This can throw a warning about float - let's hide it for now.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py

try:
    import awkward0 as awkward
except ModuleNotFoundError:
    import awkward

ja = awkward.JaggedArray

dtype_X = np.float32  ## set to float32 for use on CPU; can set to float16 for GPU
dtype_Y = np.float32  ## set to float32 for use on CPU; can set to float16 for GPU

VertexInfo = namedtuple("VertexInfo", ("x", "y", "z", "n", "cat"))

def collect_t2kde_data(
    *files,
    batch_size=1,
    dtype=np.float32,
    device=None,
    slice=None,
    **kargs,
):
    """
    This function collects data. It does not split it up. You can pass in multiple files.
    Example: collect_data('a.h5', 'b.h5')

    batch_size: The number of events per batch
    dtype: Select a different dtype (like float16)
    slice: Allow just a slice of data to be loaded
    device: The device to load onto (CPU by default)
    **kargs: Any other keyword arguments will be passed on to torch's DataLoader
    """

## these unit vectors will be used to convert the elements of 
## the ellipsoid major and minor axis vectors into vectors
    xhat = np.array([1, 0, 0])
    yhat = np.array([0, 1, 0])
    zhat = np.array([0, 0, 1])

    Xlist = []
    Ylist = []

    Xlist_ints = []
    Ylist_ints = []

    print("Loading data...")

    for XY_file in files:
        msg = f"Loaded {XY_file} in {{time:.4}} s"
        with Timer(msg), h5py.File(XY_file, mode="r") as f:
            ## [:,np.newaxis,:] makes X (a x b) --> (a x 1 x b) (axis 0, axis 1, axis 2)
            ## a is *probably* 4000 and b is *probably* N, but it could be the other
            ## way around;  check iwth .shape

## Here we read in the KDE itself plus the values of x and y where the KDE is maximal for 
## each bin of z. It appears that in the test file the original KDE values .AND. the values 
## of Xmax and Ymax have been divided by 2500. This should have been done only for the 
## KDE values, so Xmax and Ymax are re-scaled to better use the dynamic range available 
## using np.float16

## mds 200729  the KDE targets have many zeros. Learning zeros using a ratio
## mds         of predicted to target means that overestimating by a small
## mds         amount in the cost function, even adding an epsilon-like parameter## mds         there is difficult. Let's explicitly add epsilon here.
## mds         We might be able to do it equally well in the cost function,
## mds         but doing it here makes plotting easy as well.

            epsilon = 0.001 
## mds 201019            k)ernel = np.asarray(f["kernel"]) + epsilon
## we want to use the poca KDE, not the original kernel
            kernel = np.asarray(f["poca_KDE_A"]) + epsilon
            Xmax = 2500.*np.asarray(f["Xmax"])
            Ymax = 2500.*np.asarray(f["Ymax"]) 
            
            Y = ja.concatenate((kernel,Xmax,Ymax),axis=1).astype(dtype_Y)

            print("  ")
            print("kernel.shape = ",kernel.shape)
            print("kernel.shape[0] = ",kernel.shape[0])
            print("kernel.shape[1] = ",kernel.shape[1])
            print("Y.shape =      ",Y.shape)
            nEvts = kernel.shape[0]
            nBins = kernel.shape[1]
            binsPerInterval = int(100)
            nIntervals = int(nBins/binsPerInterval)
            print("binsPerInterval = ",binsPerInterval)
            print("nIntervals =       ",nIntervals)
            if (nBins != (binsPerInterval*nIntervals)):
              print("nBins = ",nBins)
              print("binsPerInterval*nIntervals = ",binsPerInterval*nIntervals)

            intervalKernels = np.reshape(kernel,(nEvts*nIntervals,binsPerInterval))
            intervalXmax    = np.reshape(Xmax,(nEvts*nIntervals,binsPerInterval))
            intervalYmax    = np.reshape(Ymax,(nEvts*nIntervals,binsPerInterval))
## don't want Xmas and Ymax            Y_intervals     = ja.concatenate((intervalKernels,intervalXmax,intervalYmax),axis=1).astype(dtype_Y)
            Y_intervals     = intervalKernels


            print("intervalKernels.shape = ",intervalKernels.shape)


##  code to test that intervalKernels is organized 'as expected'
## mds             for index in range(99):
## mds               print("index = ",index)
## mds               print("kernel[0,index], intervalKernels[0,index], Delta = ", kernel[0,index], intervalKernels[0,index], kernel[0,index]-intervalKernels[0,index])
## mds               print("kernel[0,100+index], intervalKernels[1,index], Delta = ",kernel[0,100+index]-intervalKernels[1,index])
## mds             
## now build the feature set from the relevant tracks' parameters
## we need to use "afile" to account for the variable length
## structure of the awkward arrays

##  201018  use poca ellipsoid parameter rather than "track parameters"
        
            afile = awkward.hdf5(f)

##  220715 remove pocaz scaling here to use raw values in mm
##  we probably want to maintain scales in mm everywhere
##  or consistently rescale all of x,y,z,A,B, etc.            
##            pocaz = np.asarray(0.001*afile["poca_z"].astype(dtype_Y))
            pocaz = np.asarray(afile["poca_z"].astype(dtype_Y))
            pocax = np.asarray(afile["poca_x"].astype(dtype_Y))
            pocay = np.asarray(afile["poca_y"].astype(dtype_Y))
            pocaMx = np.asarray(afile["major_axis_x"].astype(dtype_Y))
            print("pocaMx.shape = ", pocaMx.shape)
            pocaMy = np.asarray(afile["major_axis_y"].astype(dtype_Y))
            pocaMz = np.asarray(afile["major_axis_z"].astype(dtype_Y))
            poca_m1x = np.asarray(afile["minor_axis1_x"].astype(dtype_Y))
            poca_m1y = np.asarray(afile["minor_axis1_y"].astype(dtype_Y))
            poca_m1z = np.asarray(afile["minor_axis1_z"].astype(dtype_Y))
            poca_m2x = np.asarray(afile["minor_axis2_x"].astype(dtype_Y))
            poca_m2y = np.asarray(afile["minor_axis2_y"].astype(dtype_Y))
            poca_m2z = np.asarray(afile["minor_axis2_z"].astype(dtype_Y))

            nEvts = len(pocaz)
            print("nEvts = ", nEvts)
            print("pocaz.shape = ",pocaz.shape)

            print("len(pocaMx[0]) = ", len(pocaMx[0]))
            print("len(pocaMx[1]) = ", len(pocaMx[1]))
            print("len(pocaMx[2]) = ", len(pocaMx[2]))
            print("len(pocaMx[3]) = ", len(pocaMx[3]))
            print("len(pocaMx[4]) = ", len(pocaMx[4]))


##  220817 mds
##  add code to "scrub" poca_ellipsoid data to make sure that when
##  there are illegal values (that can lead to nan results later)
##  they are replaced and the corresponding tracks are "marked"
##  with pocaz values large enough that the tracks will be rejected
##  when IntervalTracks are constructed

            for iEvt in range(nEvts):
                l_pocaz = pocaz[iEvt][:]
                nTrks = l_pocaz.shape[0]
                
##  --    
                if (iEvt < 10):
                  print(" iEvt, nTrks = ", iEvt, nTrks)
##  --
            
                l_pocax = pocax[iEvt][:]
                l_pocay = pocay[iEvt][:]
                
                l_pocaMx = pocaMx[iEvt][:]   
                l_pocaMy = pocaMy[iEvt][:]
                l_pocaMz = pocaMz[iEvt][:]
                
                l_poca_m1x = poca_m1x[iEvt][:]   
                l_poca_m1y = poca_m1y[iEvt][:]
                l_poca_m1z = poca_m1z[iEvt][:]
                
                l_poca_m2x = poca_m2x[iEvt][:]   
                l_poca_m2y = poca_m2y[iEvt][:]
                l_poca_m2z = poca_m2z[iEvt][:]
                
                
                mag_1_sq = np.multiply(l_poca_m1x,l_poca_m1x)
                mag_1_sq = mag_1_sq + np.multiply(l_poca_m1y,l_poca_m1y)
                mag_1_sq = mag_1_sq + np.multiply(l_poca_m1z,l_poca_m1z)
                mag1     = np.sqrt(mag_1_sq)
                    
                mag_2_sq = np.multiply(l_poca_m2x,l_poca_m2x)
                mag_2_sq = mag_1_sq + np.multiply(l_poca_m2y,l_poca_m2y)
                mag_2_sq = mag_1_sq + np.multiply(l_poca_m2z,l_poca_m2z)
                mag2     = np.sqrt(mag_2_sq)
                
##  --    
                if (iEvt < 0):
                    maxTrk = min(5,nTrks)
                    for iTrk in range(maxTrk):
                        print(" iEvt, iTrk = ", iEvt, iTrk)
                        print("l_poca_m1(x,y,z)[iTrk], mag1 = ",l_poca_m1x[iTrk],l_poca_m1y[iTrk],l_poca_m1y[iTrk],mag1[iTrk])
##  --
                                            
                for iTrk in range(nTrks):
                    good_pocaMx = math.isfinite(l_pocaMx[iTrk])
                    bad_pocaMx = not good_pocaMx
                    good_pocaMy = math.isfinite(l_pocaMy[iTrk])
                    bad_pocaMy = not good_pocaMy
                    good_pocaMz = math.isfinite(l_pocaMz[iTrk])
                    bad_pocaMz = not good_pocaMz
                    if (mag1[iTrk]<1e-10 or mag2[iTrk]<1e-10 or
                        bad_pocaMx or bad_pocaMy or bad_pocaMz) :
                        print(" BAD ---- iEvt, iTrk = ",iEvt,iTrk)
## mds 220826                        print("l_pocaMx[iTrk] = ",l_pocaMx[iTrk])
## mds 220826                        print("l_pocaMy[iTrk] = ",l_pocaMy[iTrk])
## mds 220826                        print("l_pocaMz[iTrk] = ",l_pocaMz[iTrk])
## mds 220826                        print("l_poca_m1x[iTrk] = ",l_poca_m1x[iTrk])
## mds 220826                        print("l_poca_m1y[iTrk] = ",l_poca_m1y[iTrk])
## mds 220826                        print("l_poca_m1z[iTrk] = ",l_poca_m1z[iTrk])
## mds 220826                        print("l_poca_m2x[iTrk] = ",l_poca_m2x[iTrk])
## mds 220826                        print("l_poca_m2y[iTrk] = ",l_poca_m2y[iTrk])
## mds 220826                        print("l_poca_m2z[iTrk] = ",l_poca_m2z[iTrk])
                        
                        
## if there is a problem, over-write the error ellipsoid values
## with bogus (but finite & orthogonal) values that will flag
## the later code to ignore this track
                        pocaMx[iEvt][iTrk] = 0.0
                        pocaMy[iEvt][iTrk] = 0.0
                        pocaMz[iEvt][iTrk] = 100.
                        
                        poca_m1x[iEvt][iTrk] = 1.0
                        poca_m1y[iEvt][iTrk] = 0.0
                        poca_m1z[iEvt][iTrk] = 0.0
                        
                        poca_m2x[iEvt][iTrk] = 0.0
                        poca_m2y[iEvt][iTrk] = 1.0
                        poca_m2z[iEvt][iTrk] = 0.0

##  end of scrubbing code


            Mx = np.multiply(pocaMx.reshape(nEvts,1),xhat)
            My = np.multiply(pocaMy.reshape(nEvts,1),yhat)
            Mz = np.multiply(pocaMz.reshape(nEvts,1),zhat)
            majorAxis = Mx+My+Mz
            print("majorAxis.shape = ",majorAxis.shape)


            mx = np.multiply(poca_m1x.reshape(nEvts,1),xhat)
            my = np.multiply(poca_m1y.reshape(nEvts,1),yhat)
            mz = np.multiply(poca_m1z.reshape(nEvts,1),zhat)
            minorAxis_1 = mx+my+mz
            print("minorAxis_1.shape = ",minorAxis_1.shape)

            mx = np.multiply(poca_m2x.reshape(nEvts,1),xhat)
            my = np.multiply(poca_m2y.reshape(nEvts,1),yhat)
            mz = np.multiply(poca_m2z.reshape(nEvts,1),zhat)
            minorAxis_2 = mx+my+mz
            print("minorAxis_2.shape = ",minorAxis_1.shape)


            A, B, C, D, E, F = six_ellipsoid_parameters(majorAxis,minorAxis_1,minorAxis_2)

            print("A.shape = ",A.shape)
            for iTrk in range(2):
              print("majorAxis[iTrk][0][0] = ",majorAxis[iTrk][0][0])
              print("majorAxis[iTrk][1][0] = ",majorAxis[iTrk][1][0])
              print("majorAxis[iTrk][2][0] = ",majorAxis[iTrk][2][0])
              print("minorAxis_1[iTrk][0][0] = ",minorAxis_1[iTrk][0][0])
              print("minorAxis_1[iTrk][1][0] = ",minorAxis_1[iTrk][1][0])
              print("minorAxis_1[iTrk][2][0] = ",minorAxis_1[iTrk][2][0])
              print("minorAxis_2[iTrk][0][0] = ",minorAxis_2[iTrk][0][0])
              print("minorAxis_2[iTrk][1][0] = ",minorAxis_2[iTrk][1][0])
              print("minorAxis_2[iTrk][2][0] = ",minorAxis_2[iTrk][2][0])
              print("  ")
## mdsAA              print("A[iTrk][0] = ",A[iTrk][0])
## mdsAA              print("B[iTrk][0] = ",B[iTrk][0])
## mdsAA              print("C[iTrk][0] = ",C[iTrk][0])
## mdsAA              print("D[iTrk][0] = ",D[iTrk][0])
## mdsAA              print("E[iTrk][0] = ",E[iTrk][0])
## mdsAA              print("F[iTrk][0] = ",F[iTrk][0])
## mds              print("majorAxis[iTrk][0] = ", majorAxis[iTrk][0])
## mds              print("majorAxis[iTrk][1] = ", majorAxis[iTrk][1])
## mds              print("majorAxis[iTrk][2] = ", majorAxis[iTrk][2])


            


## add some "debugging" code to make sure I understand enumerate
##  mds 220711

            minZ = -100.
            maxZ =  300.
            intervalLength = (maxZ-minZ)/nIntervals
            print(" *** intervalLength = ",intervalLength,"   ***")

##  mark non-track data with -99 as a flag
## mds 220821            maxIntLen = 150  ## to be re-visited  mds 220712
            maxIntLen = 250  ## increased as some intervals clearly have more than 200 tracks
            padded_int_pocaz   = np.zeros((nEvts*nIntervals,maxIntLen))-99.
            padded_int_pocax   = np.zeros((nEvts*nIntervals,maxIntLen))-99.
            padded_int_pocay   = np.zeros((nEvts*nIntervals,maxIntLen))-99.
            padded_int_pocaA   = np.zeros((nEvts*nIntervals,maxIntLen))-99.
            padded_int_pocaB   = np.zeros((nEvts*nIntervals,maxIntLen))-99.
            padded_int_pocaC   = np.zeros((nEvts*nIntervals,maxIntLen))-99.
            padded_int_pocaD   = np.zeros((nEvts*nIntervals,maxIntLen))-99.
            padded_int_pocaE   = np.zeros((nEvts*nIntervals,maxIntLen))-99.
            padded_int_pocaF   = np.zeros((nEvts*nIntervals,maxIntLen))-99.

            for  eventIndex, e in enumerate(pocaz):
              if (eventIndex<1):
                print("eventIndex = ",eventIndex)
              local_pocaz = pocaz[eventIndex][:]
              local_pocax = pocax[eventIndex][:]
              local_pocay = pocay[eventIndex][:]
              local_A = A[eventIndex][:]
              local_B = B[eventIndex][:]
              local_C = C[eventIndex][:]
              local_D = D[eventIndex][:]
              local_E = E[eventIndex][:]
              local_F = F[eventIndex][:]
  
              indices = np.argsort(local_pocaz)

              ordered_pocaz = local_pocaz[indices]
              ordered_pocax = local_pocax[indices]
              ordered_pocay = local_pocay[indices]
              ordered_A     = local_A[indices]
              ordered_B     = local_B[indices]
              ordered_C     = local_C[indices]
              ordered_D     = local_D[indices]
              ordered_E     = local_E[indices]
              ordered_F     = local_F[indices]
  
              if (eventIndex<0): 
                print("len(local_pocaz) = ",len(local_pocaz))
                print("  ")
                print("local_pocaz = ",local_pocaz)
                print("ordered_pocaz = ",ordered_pocaz) 
                print("      -----------      ")
                print("local_pocax = ",local_pocax)
                print("ordered_pocax = ",ordered_pocax)
                print("  ---------------------- \n")

              for interval in range(nIntervals):
                interval_lowEdge  = minZ + interval*intervalLength
                interval_highEdge = interval_lowEdge + intervalLength 
                interval_minZ     = interval_lowEdge - 2.5
                interval_maxZ     = interval_highEdge + 2.5
                if (eventIndex<1):
                    print(" -- interval, interval_minZ, interval_maxZ = ",interval, interval_minZ, interval_maxZ)
                if (2460==eventIndex):
                    intervalRange = (ordered_pocaz>99999.) ## ugly way to remove all poca-ellipsoids in this event
                else:
                    intervalRange = (ordered_pocaz>interval_minZ) & (ordered_pocaz<interval_maxZ)
## for each interval we want the values of z shifted to be centered at the
## center of the interval
                interval_pocaz = ordered_pocaz[intervalRange] - interval_lowEdge
                interval_pocax = ordered_pocax[intervalRange]
                interval_pocay = ordered_pocay[intervalRange]
                interval_A     = ordered_A[intervalRange]
                interval_B     = ordered_B[intervalRange]
                interval_C     = ordered_C[intervalRange]
                interval_D     = ordered_D[intervalRange]
                interval_E     = ordered_E[intervalRange]
                interval_F     = ordered_F[intervalRange]

                intervalSigmaZ = np.sqrt(np.divide(1.,interval_C))
                intervalSigmaX = np.sqrt(np.divide(1.,interval_A))
                intervalSigmaY = np.sqrt(np.divide(1.,interval_B))
                xSigmas = np.divide(interval_pocax,intervalSigmaX)
                ySigmas = np.divide(interval_pocay,intervalSigmaY)

                veryGoodTracks =  (intervalSigmaZ<2.0) & (np.absolute(xSigmas)<4.0) & (np.absolute(ySigmas)<4.0)
               
                interval_pocaz = interval_pocaz[veryGoodTracks]
                interval_pocax = interval_pocax[veryGoodTracks]
                interval_pocay = interval_pocay[veryGoodTracks]
                interval_A     = interval_A[veryGoodTracks]
                interval_B     = interval_B[veryGoodTracks]
                interval_C     = interval_C[veryGoodTracks]
                interval_D     = interval_D[veryGoodTracks]
                interval_E     = interval_E[veryGoodTracks]
                interval_F     = interval_F[veryGoodTracks]

                if (eventIndex<0): 
                    print("  ")
                    if (interval<5):
                      print("eventIndex, interval = ",eventIndex, interval)
                      print("interval_pocaz = ",interval_pocaz)
                      print("             ----          ")
                      print("interval_pocax = ",interval_pocax)

## and now for all intervals for the eventIndex range
                    print("  ")
                    print("eventIndex and interval = ",eventIndex,interval) 
                    print("interval_pocaz = ",interval_pocaz)
                fillingLength = min(len(interval_pocaz),maxIntLen)
                ii = eventIndex*nIntervals + interval
                padded_int_pocaz[ii,:fillingLength] = interval_pocaz[:fillingLength].astype(dtype_Y)
                padded_int_pocax[ii,:fillingLength] = interval_pocax[:fillingLength].astype(dtype_Y)
                padded_int_pocay[ii,:fillingLength] = interval_pocay[:fillingLength].astype(dtype_Y)
                padded_int_pocaA[ii,:fillingLength] = interval_A[:fillingLength].astype(dtype_Y)
                padded_int_pocaB[ii,:fillingLength] = interval_B[:fillingLength].astype(dtype_Y)
                padded_int_pocaC[ii,:fillingLength] = interval_C[:fillingLength].astype(dtype_Y)
                padded_int_pocaD[ii,:fillingLength] = interval_D[:fillingLength].astype(dtype_Y)
                padded_int_pocaE[ii,:fillingLength] = interval_E[:fillingLength].astype(dtype_Y)
                padded_int_pocaF[ii,:fillingLength] = interval_F[:fillingLength].astype(dtype_Y)

################                

            padded_int_pocaz  = padded_int_pocaz[:,np.newaxis,:]
            padded_int_pocax  = padded_int_pocax[:,np.newaxis,:]
            padded_int_pocay  = padded_int_pocay[:,np.newaxis,:]
            padded_int_pocaA  = padded_int_pocaA[:,np.newaxis,:]
            padded_int_pocaB  = padded_int_pocaB[:,np.newaxis,:]
            padded_int_pocaC  = padded_int_pocaC[:,np.newaxis,:]
            padded_int_pocaD  = padded_int_pocaD[:,np.newaxis,:]
            padded_int_pocaE  = padded_int_pocaE[:,np.newaxis,:]
            padded_int_pocaF  = padded_int_pocaF[:,np.newaxis,:]


            X_ints = ja.concatenate((padded_int_pocaz,padded_int_pocax,padded_int_pocay,padded_int_pocaA,padded_int_pocaB,padded_int_pocaC,padded_int_pocaD,padded_int_pocaE,padded_int_pocaF),axis=1).astype(dtype_X)

            print("len(X_ints) =",len(X_ints))

            Xlist_ints.append(X_ints)
            Ylist_ints.append(Y_intervals)

            print("len(Xlist_ints) = ",len(Xlist_ints))

    X_intervals = np.concatenate(Xlist_ints, axis = 0)
    Y_intervals = np.concatenate(Ylist_ints, axis = 0)

    print(" ")
    print("  -------  ")
    print("X_intervals.shape = ",X_intervals.shape)
    print("Y_intervals.shape = ",Y_intervals.shape)

    badEvents = [123, 4484, 4511, 5575, 8120, 8747, 10651, 11956, 12010, 14817,
                 15591, 18541, 21607, 23675, 24483, 26627, 32267, 33612,
                 35735, 38504, 40219, 40520, 42757]

    intervalsPerEvent = nIntervals
    print("nEvts, nIntervals, intervalsPerEvent = ", nEvts, nIntervals, intervalsPerEvent)

    badIntervals = np.asarray([],dtype=int)
    for iEvt in badEvents:
        badInts = np.arange(iEvt*intervalsPerEvent,iEvt*intervalsPerEvent+intervalsPerEvent)
        badIntervals = np.concatenate((badIntervals,badInts),axis=0)
        
## mds 220827    print("badIntervals = ",badIntervals)

    X_intervals = np.delete(X_intervals,badIntervals,axis=0)
    Y_intervals = np.delete(Y_intervals,badIntervals,axis=0)

    print("  ------- after dropping bad events' intervals ---  ")
    print("X_intervals.shape = ",X_intervals.shape)
    print("Y_intervals.shape = ",Y_intervals.shape)
    print("  ")
    if slice:

        X_intervals = X_intervals[slice, :]
        Y_intervals = Y_intervals[slice, :]

    with Timer(start=f"Constructing {X_intervals.shape[0]} intervals dataset"):

        x_t_intervals = torch.tensor(X_intervals)
        y_t_intervals = torch.tensor(Y_intervals)

##  for debugging
        for intervalIndex in range(00):
          print("  ")
          print(" ** intervalIndex = ",intervalIndex)
          print("y_t_intervals[intervalIndex][0:100] = ")
          print(y_t_intervals[intervalIndex][0:100])
          print("  ")
          print("x_t_intervals[intervalIndex][0][0:20] = ")
          print(x_t_intervals[intervalIndex][0][0:20])
          print(" --------- ")


        if device is not None:

            x_t_intervals = x_t_intervals.to(device)
            y_t_intervals = y_t_intervals.to(device)

        dataset = TensorDataset(x_t_intervals, y_t_intervals)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kargs)

    print("x_t_intervals.shape = ",x_t_intervals.shape)
    print("x_t_intervals.shape[0] = ", x_t_intervals.shape[0])
    print("x_t_intervals.shape[1] = ", x_t_intervals.shape[1])

    print("y_t_intervals.shape = ",y_t_intervals.shape)
    print("y_t_intervals.shape[0] = ", y_t_intervals.shape[0])
    print("y_t_intervals.shape[1] = ", y_t_intervals.shape[1])

    
    
    return loader

####### -----------------

import os
def select_gpu(selection=None):
    """
    Select a GPU if availale.

    selection can be set to get a specific GPU. If left unset, it will REQUIRE that a GPU be selected by environment variable. If -1, the CPU will be selected.
    """

    if str(selection) == "-1":
        return torch.device("cpu")

    # This must be done before any API calls to Torch that touch the GPU
    if selection is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selection)

    if not torch.cuda.is_available():
        print("Selecting CPU (CUDA not available)")
        return torch.device("cpu")
    elif selection is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is *required* when running with CUDA available"
        )

    print(torch.cuda.device_count(), "available GPUs (initially using device 0):")
    for i in range(torch.cuda.device_count()):
        print(" ", i, torch.cuda.get_device_name(i))

    return torch.device("cuda:0")

def eventID(intervalNumber):
    eventNumber = int((intervalNumber)/40)
    localInterval = intervalNumber - eventNumber*40
    return eventNumber,localInterval


import torch.nn as nn 
import torch.nn.functional as F

class Model(nn.Module):
    softplus = torch.nn.Softplus()

    def __init__(self, nOut1=25, nOut2=25, nOut3=25,
                       nOut4=25, nOut5=25):
        super(Model,self).__init__()

        self.nOut1 = nOut1
        self.nOut2 = nOut2
        self.nOut3 = nOut3
        self.nOut4 = nOut4
        self.nOut5 = nOut5
       

        self.layer1 = nn.Linear(
                    in_features = 9,
                    out_features = self.nOut1,
                    bias = True)
        self.layer2 = nn.Linear(
                    in_features = self.layer1.out_features,
                    out_features = self.nOut2,
                    bias = True)
        self.layer3 = nn.Linear(
                    in_features = self.layer2.out_features,
                    out_features = self.nOut3,
                    bias = True)
        self.layer4 = nn.Linear(
                    in_features = self.layer3.out_features,
                    out_features = self.nOut4,
                    bias = True)
        self.layer5 = nn.Linear(
                    in_features = self.layer4.out_features,
                    out_features = self.nOut5,
                    bias = True)
        self.layer6 = nn.Linear(
                    in_features = self.layer5.out_features,
                    out_features = 100,
                    bias = True)
        
    def forward(self, x):
        
        leaky = nn.LeakyReLU(0.01)
        nEvts     = x.shape[0]
        nFeatures = x.shape[1]
        nTrks     = x.shape[2]
        mask = x[:,0,:] > -98.
        filt = mask.float()
        f1 = filt.unsqueeze(2)
        f2 = f1.expand(-1,-1,100)
        x = x.transpose(1,2)
        ones = torch.ones(nEvts,nFeatures,nTrks)
        x0 = x 
        x = leaky(self.layer1(x))
        x = leaky(self.layer2(x))
        x = leaky(self.layer3(x))
        x = leaky(self.layer4(x))
        x = leaky(self.layer5(x))
        x = (self.layer6(x))  
        x = self.softplus(x)
       
        x.view(nEvts,-1,100)

        x1 = torch.mul(f2,x)
        x1.view(nEvts,-1,100)
        y_prime = torch.sum(x1,dim=1)        
        y_pred = torch.mul(y_prime,0.001)
        return y_pred