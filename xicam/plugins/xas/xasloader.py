import pandas as pd
from modpkgs.memoize import memoized_property
import numpy as np
from larch import Interpreter
from larch_plugins.math.mathutils import index_of
from larch_plugins.xafs import pre_edge
mylarch = Interpreter()



def open(f):
    cls = None
    # determine which class to use somehow
    cls = BL6311Spectra
    return cls(f)


class XASSpectra(object):
    def read(self, filename, frame=None):
        """
        To be overridden - fill in self.header and self.data
        """
        raise Exception("Class has not implemented read method yet")
        #        return self




class BL6311Spectra(XASSpectra):
    def __init__(self,f):
        self.read(f)
        self.normalize()



    def read(self, filename, scan=None):
        """import scan file"""
        scandata_f = pd.read_csv(filename, sep='\t', skiprows=10)
        if not ("Counter 0" in scandata_f.columns):
            scandata_f = pd.read_csv(filename, sep='\t', skiprows=8)  # TrajScan files need 8 header lines somehow?

        if not ("Counter 0" in scandata_f.columns):
            print ("Problem with header. skipping 12 or 10 lines did not make it. Check input file.")
            return None

        self.rawdata = scandata_f
        self.scans = [self.scan(i) for i in range(self.num_scans)]
        return self

    @memoized_property
    def is_stacked(self):
        """Determine stacked or single scan file(s) and informs user"""
        return len(self.rawdata.Energy) != len(set(self.rawdata.Energy))

    @memoized_property
    def num_scans(self):
        # Number of scans in file using set to find repeating numbers in energy column
        return len(self.rawdata.Energy) / len(set(self.rawdata.Energy))

    @memoized_property
    def num_bins(self):
        return len(set(self.rawdata.Energy))

    def scan(self, N, scan_rng=0, unstacked=pd.DataFrame(np.array([]))):
        return self.rawdata.iloc[N*self.num_bins:(N+1)*self.num_bins, :].copy()

    def normalize(self, datacounter="Counter 3", reference_counter='Counter 2'):
        """Preparing Scan (normalization by I0)"""
        if 'Counter 4' in self.rawdata.columns:
            clockname = 'Counter 4'
        elif 'Counter 6' in self.rawdata.columns:
            clockname = 'Counter 6'
        else:
            print("No counter for clock found (looked for 'Counter 4' and 'Counter 6'). Defaulting to 'Counter 0'.")
            clockname = 'Counter 0'



        for scan in self.scans:
            scan["I_Norm0"] = scan[datacounter].astype(float) / scan[reference_counter].astype(float)
            scan["I_Normt"] = scan[datacounter].astype(float) / scan[clockname].astype(float)
            scan["Energy"] = scan["Energy"].round(1)
            scan.mu = scan.I_Norm0
            scan.x = scan.Energy

            # do pre-processing steps, here XAFS pre-edge removal
            # print 'leveling'

            # pre_edge function automatically finds edge value (e0)
            # but I manually set points for leveling at -2 and -10 eV from the pre-edge.
            pre_edge(scan.x, scan.mu, pre1=-10, pre2=-2, group=scan, _larch=mylarch)
            # print 'Automatic edge position found as', scan.e0, 'eV'

            scan.y = scan.mu - scan.pre_edge  # Subtract fitted pre-edge curve
            # mdat[str(idx)] = mdat.y  # Save leveled data in matrix for later
            # select data range.
            lowcut = 50

            scan.e = scan.x[lowcut:]
            scan.y = scan.y[lowcut:]




def tests():
    s = open('/home/rp/Code/XAS/TrajScan21930.txt')
    print s.rawdata
    print s.scan(0)
    for scan in s.scans: print np.array(scan.e),np.array(scan.y)



if __name__ == '__main__':
    tests()