# --coding: utf-8 --
#! /usr/bin/env python

from collections import OrderedDict
from xicam import debugtools


################ Rules ###############
# Rule failure represents an exclusion

iseven = lambda m: not(m % 2)
is6multiple = lambda m: not(m%6) and m!=0       # TODO: check that the m!=0 condition is correct
is4multiple = lambda m: not(m%4) and m!=0
is3multiple = lambda m: not(m%3) and m!=0
is2multiple = lambda m: not(m%2) and m!=0

h = lambda h,k,l: iseven(h)
k = lambda h,k,l: iseven(k)
l = lambda h,k,l: iseven(l)
hpk = lambda h,k,l: iseven(h+k)
hpl = lambda h,k,l: iseven(h+l)
kpl = lambda h,k,l: iseven(k+l)
hpkpl = lambda h,k,l: iseven(h+k+l)
kpl4n = lambda h,k,l: is4multiple(k+l)
hpl4n = lambda h,k,l: is4multiple(h+l)
hpk4n = lambda h,k,l: is4multiple(h+k)
h4n = lambda h,k,l: is4multiple(h)
k4n = lambda h,k,l: is4multiple(k)
l4n = lambda h,k,l: is4multiple(l)
twohpl4n = lambda h,k,l: is4multiple(2*h+l)
l3n = lambda h,k,l: is3multiple(l)
l6n = lambda h,k,l: is6multiple(l)
mhpkpl3n = lambda h,k,l: is3multiple(-h+k+l)
hpl3n = lambda h,k,l: is3multiple(h+l)
hmkpl3n = lambda h,k,l: is3multiple(h-k+l)
mhpl3n = lambda h,k,l: is3multiple(-h+l)
h2n = lambda h,k,l: is2multiple(h)
k2n = lambda h,k,l: is2multiple(k)
l2n = lambda h,k,l: is2multiple(l)
kpl2n = lambda h,k,l: is2multiple(k+l)




######################################

cache = dict()


def arezeros(*args):
    if len(args)==4:
        zh,zk,zl=args[:3]
        mh,mk,ml=args[3]
        if (bool(mh) and zh) or (bool(mk) and zk) or (bool(ml) and zl):
            return False
        return True

    elif len(args)==5:
        zh,zk,zi,zl=args[:4]
        mh,mk,mi,ml=args[4]
        if (bool(mh) and zh) or (bool(mk) and zk) or (bool(mi) and zi) or (bool(ml) and zl):
            return False
        return True


class SGClass:
    conditions = OrderedDict([])

    def check(self,m,SG):
        """
        Get the relevant exclusion rule column and then test against it.
        """
        col,m=self.getcolumn(m)
        if type(self.conditions[SG]) is not list: # ALLOW LINKED CONDITIONS
            SG = self.conditions[SG]
        return self.checkcolumn(SG,col,m)

    def getcolumn(self,m):
        """
        Determine which column of the exclusion table to test with given miller indices 'm'.
        Unique SGClass subclasses may override this if there are more columns or the column conditions are unique.
        This virtual class is overridden by each crystal system.
        """
        pass

    def checkcolumn(self,SG,col,m):
        """
        Check the exclusion rules in column 'col' with miller indices 'm'.
        """
        if col is None:         # None means no column applies.
            return True
        if len(m)==4:
            m=(m[0],m[1],m[3])

        SGconditions = self.conditions[SG][col]
        if type(SGconditions) is not list:
            SGconditions = [SGconditions]
        for condition in SGconditions:
            if condition is not None:
                if not condition(*m):   # Do not change this to 'return not condition(*m)'!
                    return False
        return True

class Triclinic(SGClass):
    conditions = OrderedDict([('P1',[]),
                              (u'P1\u0305',[])])
    def check(self,m,SG):
        return True

class Monoclinic(SGClass):
    ############## IMPORTANT NOTE ################
    # Monoclinic must have the unique axis as b for this to work!
    # Status: COMPLETE!
    conditions = OrderedDict([('P2',[None,None,None]), #3
                              (u'P2₁',[None,None,k]), #4
                              ('C2',[hpk,h,k]), #5
                              ('A2',[kpl,l,k]),
                              ('I2',[hpkpl,hpl,k]),
                              ('Pm',[None,None,None]),#6
                              ('Pc',[None,l,None]),#7
                              ('Pa',[None,h,None]),
                              ('Pn',[None,hpl,None]),
                              ('Cm',[hpk,h,k]),#8
                              ('Am',[hpk,h,k]),
                              ('Im',[hpk,h,k]),
                              ('Cc',[hpk,[h,l],k]),#9
                              ('An',[kpl,[h,l],k]),
                              ('Ia',[hpkpl,[h,l],k]),
                              ('P2/m',[None,None,None]),#10
                              (u'P2₁/m',[None,None,k]),#11
                              ('C2/m',[hpk,h,k]),#12
                              ('A2/m',[kpl,l,k]),
                              ('I2/m',[hpkpl,hpl,k]),
                              ('P2/c',[None,l,None]),#13
                              ('P2/a',[None,h,None]),
                              ('P2/n',[None,hpl,None]),
                              (u'P2₁/c',[None,l,k]),#14
                              (u'P2₁/a',[None,h,k]),
                              (u'P2₁/n',[None,hpl,k]),
                              ('C2/c',[hpk,[h,l],k]),#15
                              ('A2/n',[kpl,[h,l],k]),
                              ('I2/a',[hpkpl,[h,l],k])])



    def getcolumn(self,m):
        if arezeros(1,0,1,m):
            column = 2
        elif arezeros(0,1,0,m) or arezeros(1,1,0,m) or arezeros(0,1,1,m):
            column = 1
        elif arezeros(0,0,0,m) or arezeros(1,0,0,m) or arezeros(0,0,1,m):   # TODO: why doesn't this column just say hkl?
            column = 0
        else:
            debugtools.frustration()
            raise ValueError
        return column,m

class Orthorhombic(SGClass):
    # Status: COMPLETE!
    conditions = OrderedDict([('P222',[None,None,None,None,None,None,None]), #16
                              (u'P222₁',[None,None,None,None,None,None,l]),#17
                              (u'P22₁2',[None,None,None,None,None,k,None]),
                              (u'P2₁22',[None,None,None,None,h,None,None]),
                              (u'P2₁2₁2',[None,None,None,None,h,k,None]),#18
                              (u'P22₁2₁',[None,None,None,None,None,k,l]),
                              (u'P2₁22₁',[None,None,None,None,h,None,l]),
                              (u'P2₁2₁2₁',[None,None,None,None,h,k,l]),#19
                              (u'C222₁',[hpk,k,h,hpk,h,k,l]),#20,unfilled to 25
                              ('C222',[hpk,k,h,hpk,h,k,None]),#21
                              ('F222',[[hpk,hpl,kpl],[k,l],[h,l],[h,k],h,k,l]),#22
                              ('I222',[hpkpl,kpl,hpl,hpk,h,k,l]),#23
                              (u'I2₁2₁2₁',[hpkpl,kpl,hpl,hpk,h,k,l]),#24
                              ('Pmm2',[None,None,None,None,None,None,None]),#25
                              ('Pm2m',[None,None,None,None,None,None,None]),
                              ('P2mm',[None,None,None,None,None,None,None]),
                              (u'Pmc2₁',[None,None,l,None,None,None,l]),#26
                              (u'P2₁ma',[None,None,None,h,h,None,None]),
                              (u'Pm2₁b',[None,None,None,k,None,k,None]),
                              (u'P2₁am',[None,None,h,None,h,None,None]),
                              ('Pcc2',[None,l,l,None,None,None,l]),#27
                              ('P2aa',[None,None,h,h,h,None,None]),
                              ('Pma2',[None,None,h,None,h,None,None]),#28
                              ('Pm2a',[None,None,None,h,h,None,None]),
                              ('P2mb',[None,None,None,h,None,k,None]),
                              ('P2cm',[None,None,l,None,None,None,l]),
                              (u'Pca2₁',[None,l,h,None,h,None,l]),#29
                              (u'P2₁ab',[None,None,h,k,h,k,None]),
                              (u'P2₁ca',[None,None,h,None,h,None,l]),
                              ('Pnc2',[None,kpl,l,None,None,k,l]),
                              (u'Pmn2₁',[None,None,hpl,None,h,None,l]), #31
                              (u'Pmn2₁',[None,None,hpl,None,h,None,l]),
                              (u'P2₁nm', [None, None, None, None, None, None, None]),
                              (u'Pmnm', [None, None, None, None, None, None, None]),
                              (u'P2na', [None, None, hpl, h, h, None, l]),
                              (u'Pmna', [None, None, hpl, h, h, None, l]),
                              (u'P2na', [None, None, hpl, k, h, k, l]),
                              (u'P2nm', [None, None, hpl, hpk, h, k, l]),
                              (u'Pmnn', [None, None, hpl, hpk, h, k, l]),
                              (u'Pbm2', [None, k,None,None,None,k,None]),
                              (u'Pb2₁m', [None, None, None, None, None, None, None]),
                              (u'Pbmm', [None, None, None, None, None, None, None]),
                              (u'Pb2b', [None, k, None, k, None, k, None]),
                              (u'Pbmb', [None, k, None, k, None, k, None]),
                              (u'Pb2n', [None, k, None, hpk, h, k, None]),
                              (u'Pbmn', [None, k, None, hpk, h, k, None]),
                              (u'Pba2', [None, k, h, None, k, h, None]),
                              (u'Pbam', [None, k, h, None, k, h, None]),
                              (u'Paa',  [None, k, h, h, h, k, None]),
                              (u'Pbab', [None, k, h, k, h, k, None]),
                              (u'Pan',  [None, k, h, hpk, h, k, None]),
                              (u'Pbc2₁', [None, k, l, None, None, k, l]),
                              (u'Pbcm',  [None, k, l, None, None, k, l]),
                              (u'Pbca',  [None, k, l, h, h, k, l]),
                              (u'Pbcb',  [None,k, l, k, None, k, l]),
                              (u'Pbcn',  [None, k, l, hpk, h, k, l]),
                              (u'Pbn2₁', [None, k, hpl, None, h, k, l]),
                              (u'Pbnm',  [None, k, hpl, None, h, k, l]),
                              (u'Pbna', [None, k, hpl, h, h, k, l]),
                              (u'Pbnb', [None, k, hpl, k, h, k, l]),
                              (u'Pbnn', [None, k, hpl, hpk, h, k, l]),
                              (u'Pcm2₁', [None, l, None, None, None, None, l]),
                              (u'Pc2m', [None, None, None, None, None, None, None]),
                              (u'Pcmm', [None, None, None, None, None, None, None]),
                              (u'Pc2a', [None, l, None, h, h, None, l]),
                              (u'Pcma', [None, l, None, h, h, None, l]),
                              (u'Pc2₁b', [None, l, None, k, None, k, l]),
                              (u'Pcmb',  [None, l, None, k, None, k, l]),
                              (u'Pc2₁n', [None, l, None, hpk, h, k, l]),
                              (u'Pcmn',  [None, l, None, hpk, h, k, l]),
                              (u'Pca2₁', [None, l, h, None, h, None, l]),
                              (u'Pcam',  [None, l, h, None, h, None, l]),
                              (u'Pcaa', [None, l, h, h, h, None, l]),
                              (u'Pcab', [None, l, h, k, h, k, l]),
                              (u'Pcan', [None, l, h, hpk, h, k, l]),
                              (u'Pcc2', [None, l, h, None, None, None, l]),
                              (u'Pccm', [None, l, l, None, None, None, l]),
                              (u'Pcca', [None, l, l, h, h, None, l]),
                              (u'Pccb', [None, l, l, k, None, k, l]),
                              (u'Pccn', [None, l, l, hpk, h, k, l]),
                              (u'Pcn2', [None, l, hpl, None, h, None, l]),
                              (u'Pcnm', [None, l, hpl, None, h, None, l]),
                              (u'Pcna', [None, l, hpl, h, h, None, l]),
                              (u'Pcnb', [None, l, hpl, k, h, k, l]),
                              (u'Pcnn', [None, l, hpl, hpk, h, k, l]),
                              (u'Pnm2₁', [None, kpl, None, None, None, k, l]),
                              (u'Pnmm', [None, kpl, None, None, None, k, l]),
                              (u'Pn2₁m', [None, None, None, None, None, None, None]),
                              (u'Pn2₁a', [None, kpl, None, h, h, k, l]),
                              (u'Pnma',  [None, kpl, None, h, h, k, l]),
                              (u'Pn2b',  [None, kpl, None, k, None, k, l]),
                              (u'Pnmb',  [None, kpl, None, k, None, k, l]),
                              (u'Pn2n', [None, kpl, None, hpk, h, k, l]),
                              (u'Pnmn', [None, kpl, None, hpk, h, k, l]),
                              (u'Pna2₁', [None, kpl, h, None, h, k, l]),
                              (u'Pnam',  [None, kpl, h, None, h, k, l]),
                              (u'Pnaa', [None, kpl, h, h, h, k, l]),
                              (u'Pnab', [None, kpl, h, k, h, k, l]),
                              (u'Pnan', [None, kpl, h, hpk, h, k, l]),
                              (u'Pnc2', [None, kpl, l, None, None, k, l]),
                              (u'Pncm', [None, kpl, l, None, None, k, l]),
                              (u'Pnca', [None, kpl, l, h, h, k, l]),
                              ('Pncb', [None,kpl,l,k,None,k,l]),
                              ('Pncn', [None, kpl, l,hpk,h,k,l]),
                              ('Pnn2', [None, kpl, hpl,None,h,k,l]),
                              ('Pnnm', [None, kpl, hpl,None,h,k,l]),
                              ('Pnna', [None, kpl, hpl,h,h,k,l]),
                              ('Pnnb', [None, kpl, hpl,k,h,k,l]),
                              ('Pnnn', [None, kpl, hpl,hpk,h,k,l]),
                              ('C222', [hpk, k, h,hpk,h,k,None]),
                              ('Cmm2', [hpk, k, h,hpk,h,k,None]),
                              ('Cmmm', [hpk, k, h,hpk,h,k,None]),
                              ('Cm2m', [hpk, k, h,hpk,h,k,None]),
                              ('C2mm', [hpk, k, h,hpk,h,k,None]),
                              (u'C222\u2081',[hpk, k, h,hpk,h,k,l]),
                              ('Cm2e', [hpk, k, h,[h,k],h,k,None]),
                              ('Cmme', [hpk, k, h,[h,k],h,k,None]),
                              ('C2me', [hpk, k, h,[h,k],h,k,None]),
                              (u'Cm2\u2081', [hpk, k, [h,l],hpk,h,k,l]),
                              ('Cmcm', [hpk, k, [h,l],hpk,h,k,l]),
                              ('C2cm', [hpk, k, [h,l],hpk,h,k,l]),
                              ('C2ce', [hpk,k,[h,l],[h,k],h,k,l]),
                              ('Cmce',   [hpk,k,[h,l],[h,k],h,k,l]),
                              (u'Ccm2\u2081', [hpk,[k,l],h,hpk,h,k,l]),
                              ('Ccmm', [hpk,[k,l],h,hpk,h,k,l]),
                              ('Cc2m', [hpk,[k,l],h,hpk,h,k,l]),
                              ('Cc2e', [hpk,[k,l],h,[h,k],h,k,l]),
                              ('Ccme', [hpk,[k,l],h,[h,k],h,k,l]),
                              ('Ccc2', [hpk,[k,l],[h,l],hpk,h,k,l]),
                              ('Cccm', [hpk,[k,l],[h,l],hpk,h,k,l]),
                              ('Ccce', [hpk,[k,l],[h,l],[h,k],h,k,l]),
                              ('B222', [hpl,l,hpl,h,h,None,l]),
                              ('Bmm2', [hpl,l,hpl,h,h,None,l]),
                              ('Bmmm', [hpl,l,hpl,h,h,None,l]),
                              ('Bm2m', [hpl,l,hpl,h,h,None,l]),
                              ('B2mm',   [hpl,l,hpl,h,h,None,l]),
                              (u'B22\u20812', [hpl,l,hpl,h,h,k,l]),
                              (u'Bm2\u2081b', [hpl,l,hpl,[h,k],h,k,l]),
                              ('Bmmb', [hpl,l,hpl,[h,k],h,k,l]),
                              ('B2mb', [hpl,l,hpl,[h,k],h,k,l]),
                              ('Bm2e', [hpl,l,[h,l],h,h,None,l]),
                              ('Bmem', [hpl,l,[h,l],h,h,None,l]),
                              ('B2em', [hpl,l,[h,l],h,h,None,l]),
                              ('B2eb', [hpl,l,[h,l],[h,k],h,k,l]),
                              ('Bmeb', [hpl,l,[h,l],[h,k],h,k,l]),
                              ('Bbm2', [hpl,[k,l],hpl,h,h,k,l]),
                              ('Bbmm', [hpl,[k,l],hpl,h,h,k,l]),
                              (u'Bb2\u2081m', [hpl,[k,l],hpl,h,h,k,l]),
                              ('Bb2b', [hpl,[k,l],hpl,[h,k],h,k,l]),
                              ('Bbmb', [hpl,[k,l],hpl,[h,k],h,k,l]),
                              ('Bbe2', [hpl,[k,l],[h,l],h,h,k,l]),
                              ('Bbem', [hpl,[k,l],[h,l],h,h,k,l]),
                              ('Bbeb', [hpl,[k,l],[h,l],[h,k],h,k,l]),
                              ('A222', [kpl,kpl,l,k,None,k,l]),
                              ('Amm2', [kpl,kpl,l,k,None,k,l]),
                              ('Ammm', [kpl,kpl,l,k,None,k,l]),
                              ('Am2m', [kpl,kpl,l,k,None,k,l]),
                              ('A2mm', [kpl,kpl,l,k,None,k,l]),
                              (u'A2\u208122', [kpl,kpl,l,k,h,k,l]),
                              ('Am2a', [kpl,kpl,l,[h,k],h,k,l]),
                              ('Amma', [kpl,kpl,l,[h,k],h,k,l]),
                              (u'A2\u2081ma', [kpl,kpl,l,[h,k],h,k,l]),
                              ('Ama2', [kpl,kpl,[h,l],k,h,k,l]),
                              ('Amam', [kpl,kpl,[h,l],k,h,k,l]),
                              ('A2aa', [kpl,kpl,[h,l],[h,k],h,k,l]),
                              ('Amaa', [kpl,kpl,[h,l],[h,k],h,k,l]),
                              ('Aem2', [kpl,[k,l],l,k,None,k,l]),
                              ('Aemm', [kpl,[k,l],l,k,None,k,l]),
                              ('Ae2m', [kpl,[k,l],l,k,None,k,l]),
                              ('Ae2a', [kpl,[k,l],l,[h,k],h,k,l]),
                              ('Aema', [kpl,[k,l],l,[h,k],h,k,l]),
                              ('Aea2', [kpl,[k,l],[h,l],k,h,k,l]),
                              ('Aeam', [kpl,[k,l],[h,l],k,h,k,l]),
                              ('Aeaa', [kpl,[k,l],[h,l],[h,k],h,k,l]),
                              ('I222', [hpkpl,kpl,hpl,hpk,h,k,l]),
                              ('Imm2', [hpkpl,kpl,hpl,hpk,h,k,l]),
                              ('Immm', [hpkpl,kpl,hpl,hpk,h,k,l]),
                              (u'I2\u20812\u20812\u2081', [hpkpl,kpl,hpl,hpk,h,k,l]),
                              ('Im2m', [hpkpl,kpl,hpl,hpk,h,k,l]),
                              ('F222', [[hpk,hpl,kpl],[k,l],[h,l],[h,k],h,k,l]),
                              ('I2mm', [hpkpl,kpl,hpl,hpk,h,k,l]),
                              ('Im2a',[hpkpl,kpl,hpl,[h,k],h,k,l]),
                              ('I2mb',[hpkpl,kpl,hpl,[h,k],h,k,l]),
                              ('Ima2',[hpkpl,kpl,[h,l],hpk,h,k,l]),
                              ('I2cm',[hpkpl,kpl,[h,l],hpk,h,k,l]),
                              ('I2cb',[hpkpl,kpl,[h,l],[h,k],h,k,l]),
                              ('Iem2',[hpkpl,kpl,[h,l],[h,k],h,k,l]),
                              ('Ic2a',[hpkpl,[k,l],hpl,[h,k],h,k,l]),
                              ('Iba2',[hpkpl,[k,l],[h,l],hpk,h,k,l]),
                              ('Fmm2',[[hpk,hpl,kpl],[k,l],[h,l],[h,k],h,k,l]),
                              ('Fm2m',[[hpk,hpl,kpl],[k,l],[h,l],[h,k],h,k,l]),
                              ('F2mm',[[hpk,hpl,kpl],[k,l],[h,l],[h,k],h,k,l]),
                              ('F2dd',[[hpk,hpl,kpl],[k,l],[hpl4n,h,l],[hpk4n,h,k],h4n,k4n,l4n]),
                              ('Fd2d',[[hpk,hpl,kpl],[kpl4n,k,l],[h,l],[hpk4n,h,k],h4n,k4n,l4n]),
                              ('Fdd2',[[hpk,hpl,kpl],[kpl4n,k,l],[hpl4n,h,l],[h,k],h4n,k4n,l4n]),
                              ('Imma',[hpkpl,kpl,hpl,[h,k],h,k,l]),
                              ('Immb',[hpkpl,kpl,hpl,[h,k],h,k,l]),
                              ('Imam',[hpkpl,kpl,[h,l],hpk,h,k,l]),
                              ('Imcm',[hpkpl,kpl,[h,l],hpk,h,k,l]),
                              ('Imcb',[hpkpl,kpl,[h,l],[h,k],h,k,l]),
                              ('Iemm',[hpkpl,kpl,[h,l],[h,k],h,k,l]),
                              ('Icma',[hpkpl,[k,l],hpl,[h,k],h,k,l]),
                              ('Ibam',[hpkpl,[k,l],[h,l],hpk,h,k,l]),
                              ('Ibca',[hpkpl,[k,l],[h,l],[h,k],h,k,l]),
                              ('Icab',[hpkpl,[k,l],[h,l],[h,k],h,k,l]),
                              ('Fmmm',[[hpk,hpl,kpl],[k,l],[h,l],[h,k],h,k,l]),
                              ('Fddd',[[hpk,hpl,kpl],[kpl4n,k,l],[hpl4n,h,l],[hpk4n,h,k],h4n,k4n,l4n])
    ])

    def getcolumn(self,m):
        if arezeros(1,1,0,m):
            column = 6
        elif arezeros(1,0,1,m):
            column = 5
        elif arezeros(0,1,1,m):
            column = 4
        elif arezeros(0,0,1,m):
            column = 3
        elif arezeros(0,1,0,m):
            column = 2
        elif arezeros(1,0,0,m):
            column = 1
        else:
            column = 0
        return column,m

class Tetragonal(SGClass):
    #Status: COMPLETE!

    conditions = OrderedDict([('P4',[None,None,None,None,None,None,None]),
                              (u'P4\u0305','P4'),
                              ('P4/m','P4'),
                              ('P422','P4'),
                              ('P4mm','P4'),
                              (u'P4\u03052m','P4'),
                              ('P4/mmm','P4'),
                              (u'P4\u0305m2','P4'),
                              (u'P42₁2',[None,None,None,None,None,k,None]),
                              (u'P4\u03052₁m',u'P42₁2'),
                              (u'P4₂',[None,None,None,None,l,None,None]),
                              (u'P4₂/m',u'P4₂'),
                              (u'P4₂22',u'P4₂'),
                              (u'P4₂2₁2',[None,None,None,None,l,k,None]),
                              (u'P4₁',[None,None,None,None,l4n,None,None]),
                              (u'P4₃',u'P4₁'),
                              (u'P4₁22',u'P4₁'),
                              (u'P4₃22',u'P4₁'),
                              (u'P4₁2₁2',[None,None,None,None,l4n,k,None]),
                              (u'P4₃2₁2',u'P4₁2₁2'),
                              (u'P4₂mc',[None,None,None,l,l,None,None]),
                              (u'P4\u03052c',u'P4₂mc'),
                              (u'P42/mmc',u'P4₂mc'),
                              (u'P4\u03052₁c',[None,None,None,l,l,k,None]),
                              ('P4bm',[None,None,k,None,None,k,None]),
                              (u'P4\u0305b2','P4bm'),
                              (u'P4/mbm','P4bm'),
                              (u'P4₂bc',[None,None,k,l,l,k,None]),
                              (u'P4₂/mbc',u'P4₂bc'),
                              (u'P4₂cm',[None,None,l,None,l,None,None]),
                              (u'P4\u0305c2',u'P4₂cm'),
                              (u'P4₂/mcm',u'P4₂cm'),
                              (u'P4cc',[None,None,l,l,l,None,None]),
                              (u'P4/mcc',u'P4cc'),
                              (u'P4₂nm',[None,None,kpl,None,l,k,None]),
                              (u'P4\u0305n2',u'P4₂nm'),
                              (u'P4₂/mnm',u'P4₂nm'),
                              (u'P4nc',[None,None,kpl,l,l,k,None]),
                              (u'P4/mnc',u'P4nc'),
                              (u'P4/n',[None,hpk,None,None,None,k,None]),
                              (u'P4/nmm',u'P4/n'),
                              (u'P4₂/n',[None,hpk,None,None,l,k]),
                              (u'P4₂/nmc',[None,hpk,None,l,l,k,None]),
                              (u'P4/nbm',[None,hpk,k,None,None,k,None]),
                              (u'P4₂/nbc',[None,hpk,k,l,l,k,None]),
                              (u'P4₂/ncm',[None,hpk,l,None,l,k,None]),
                              (u'P4/ncc',[None,hpk,l,l,l,k,None]),
                              (u'P4₂nnm',[None,hpk,kpl,None,l,k,None]),
                              (u'P4/nnc',[None,hpk,kpl,l,l,k,None]),
                              (u'I4',[hpkpl,hpk,kpl,l,l,k,None]),
                              (u'I4\u0305',u'I4'),
                              (u'I4/m',u'I4'),
                              (u'I422',u'I4'),
                              (u'I4mm',u'I4'),
                              (u'I4\u03052m',u'I4'),
                              (u'I4/mmm',u'I4'),
                              (u'I4\u0305m2',u'I4'),
                              (u'I4₁',[hpkpl,hpk,kpl,l,l4n,k,None]),
                              (u'I4₁22',u'I4₁'),
                              (u'I4₁md',[hpkpl,hpk,kpl,twohpl4n,l4n,k,h]),
                              (u'I4\u03052d',u'I4₁md'),
                              (u'I4cm',[hpk,l,hpk,[k,l],l,l,k,None]),
                              (u'I4\u0305c2',u'I4cm'),
                              (u'I4/mcm',u'I4cm'),
                              (u'I4₁cd',[hpkpl,hpk,[k,l],twohpl4n,l4n,k,h]),
                              (u'I4₁/a',[hpkpl,[h,k],kpl,l,l4n,k,None]),
                              (u'I4₁/amd',[hpkpl,[h,k],kpl,twohpl4n,l4n,k,h]),
                              (u'I4₁/acd',[hpkpl,[h,k],[k,l],twohpl4n,l4n,k,h])
                              ])

    def getcolumn(self,m):
        mh,mk,ml=m
        if mh==mk and arezeros(0,0,1,m):
            column = 6
        elif arezeros(1,0,1,m):
            column = 5
        elif arezeros(1,1,0,m):
            column = 4
        elif mh==mk:          # TODO: is is possible that no column condition is satisfied (i.e. 100). What is supposed to happen then? if the intention would be to apply column 0, why does monoclinic have hkl 0kl h0l in column 0?
            column = 3
        elif arezeros(1,0,0,m):
            column = 2
        elif arezeros(0,0,1,m):
            column = 1
        else:
            column = 0
        return column,m

class Trigonal(SGClass):
    # Status: COMPLETE!
    ## Hexagonal miller axes disabled to eliminate conflicts
    #
    # conditions = OrderedDict([('P3',[None, None, None, None]),
    #                             (u'P3\u0305',[None, None, None, None]),
    #                             ('P321',[None, None, None, None]),
    #                             ('P3m1',[None, None, None, None]),
    #                             (u'P3m\u03051',[None, None, None, None]),
    #                             ('P312',[None, None, None, None]),
    #                             ('P31m',[None, None, None, None]),
    #                             (u'P3\u03051m',[None, None, None, None]),
    #                             (u'p3₁',[None, None, None, l3n]),
    #                             (u'p3₁21',[None, None, None, l3n]),
    #                             ('p3₁12',[None, None, None, l3n]),
    #                             ('p3₂',[None, None, None, l3n]),
    #                             (u'p3₂21',[None, None, None, l3n]),
    #                             (u'p3₂12',[None, None, None, l3n]),
    #                             ('P31c',[None, None, l, l]),
    #                             (u'P3\u03051c',[None, None, l, l]),
    #                             ('P3c1',[None, l, None, l]),
    #                             (u'P3\u0305c1',[None, l, None, l]),
    #                             ('R3',[mhpkpl3n, hpl3n, l3n, l3n]),     # TODO: Why are there duplicate entries in the SG table for the R3's?
    #                             (u'R3\u0305',[mhpkpl3n, hpl3n, l3n, l3n]),
    #                             ('R32',[mhpkpl3n, hpl3n, l3n, l3n]),
    #                             ('R3m',[mhpkpl3n, hpl3n, l3n, l3n]),
    #                             (u'R3\u0305m',[mhpkpl3n, hpl3n, l3n, l3n]),
    #                             ('R3c',[mhpkpl3n, [hpl3n,l], l3n, l6n]),
    #                             (u'R3\u0305c',[mhpkpl3n, [hpl3n,l], l3n, l6n]),
    #                             ('R3c',[mhpkpl3n, [hpl3n,l], l3n, l6n]),
    #                           ])

    conditions = OrderedDict([('R3',[None,None,None,None]),
                              (u'R3\u0305','R3'),
                              (u'R32','R3'),
                              (u'R3m','R3'),
                              (u'R3\u0305m','R3'),
                              ('R3c',[None,None,l,h]),
                              (u'R3\u0305c','R3c')])

    def getcolumn(self,m):
        if len(m)==4:
            mh,mk,mi,ml = m
            if arezeros(1,1,1,0,m):
                column = 3
            elif mh==mk and mi==-2*mh:
                column = 2
            elif mh==-mk and arezeros(0,0,1,0,m):
                column = 1
            else:
                column = 0
        elif len(m)==3:
            mh,mk,ml = m
            if mh==mk==ml:
                column = 3
            elif mh==mk:
                column = 2
            else:
                column = 1
        else:
            debugtools.frustration()
            raise ValueError
        return column,m

class Hexagonal(SGClass):
    # Status: COMPLETE!
    conditions = OrderedDict([('P6',[None, None, None]),
                                (u'P6\u0305',[None, None, None]),
                                ('P6/m',[None, None, None]),
                                ('P622',[None, None, None]),
                                ('P6mm',[None, None, None]),
                                (u'P6\u03052m',[None, None, None]),
                                ('P6/mmm',[None, None, None]),
                                (u'P6\u0305m2',[None, None, None]),
                                ('P6₃', [None, None, l]),
                                ('P6₃/m', [None, None, l]),
                                ('P6₃22', [None, None, l]),
                                ('P6₂',[None, None, l3n]),
                                ('P6₂22',[None, None, l3n]),
                                ('P6₄',[None, None, None, l3n]),
                                ('P6₁',[None, None, l6n]),
                                ('P6₁22',[None, None, l6n]),
                                ('P6₃',[None, None, l6n]),
                                ('P6₅22',[None, None, l6n]),
                                ('P6₃mc',[None, l, l]),
                                (u'P6\u03052c',[None, l, l]),
                                ('P6₃/mmc',[None, l, l]),
                                ('P6₃cm',[l, None, l]),
                                (u'P6\u0305c2',[l, None, l]),
                                ('P6₃/mcm',[l, None, l]),
                                ('P6cc',[l,l,l]),
                                ('P6/mcc',[l,l,l])
                              ])

    def getcolumn(self,m):
        mh,mk,ml = m
        mi = -(mh+mk)
        hexm = (mh,mk,mi,ml)
        if arezeros(1,1,1,0,hexm):
            column = 2
        elif mh==mk and mi==-2*mh:          # TODO: What rules apply to hexagonal 1111
            column = 1
        elif arezeros(0,0,1,0,hexm) and mh==-mk:
            column = 0
        else:
            column = None
        return column,m

class Cubic(SGClass):
    # Status: COMPLETE
    conditions = OrderedDict([('P23',[None, None, None, None]),
                                (u'Pm3\u0305',[None, None, None, None]),
                                ('P432',[None, None, None, None]),
                                (u'P4\u03053m',[None, None, None, None]),
                                (u'Pm3\u0305m',[None, None, None, None]),
                                ('P2₁3',[None, None, None, l]),
                                ('P4₂32',[None, None, None, l]),
                                ('P4₁32',[None, None, None, l4n]),
                                ('P4₃32',[None, None, None, l4n]),
                                (u'P4\u03053n',[None, None, l, l]),
                                (u'Pm3\u0305n',[None, None, l, l]),
                                (u'Pa3\u0305',[None, k, None, l]), # TODO: Handle this special case!
                                (u'Pn3\u0305',[None, kpl, None, l]),
                                (u'Pn3\u0305m',[None, kpl, None, l]),
                                (u'Pn3\u0305n',[None, kpl, l, l]),
                                ('I23', [hpkpl, kpl, l, l]),
                                ('I2₁3', [hpkpl, kpl, l, l]),
                                (u'Im3\u0305', [hpkpl, kpl, l, l]),
                                ('I432', [hpkpl, kpl, l, l]),
                                (u'I4\u03053m', [hpkpl, kpl, l, l]),
                                (u'Im3\u0305m (BCC)', [hpkpl, kpl, l, l]),
                                ('I4₁32',[hpkpl, kpl, l, l4n]),
                                (u'I4\u03053d',[hpkpl, kpl, [twohpl4n,l],l4n]),
                                (u'Ia3\u0305',[hpkpl, [k,l], l, l]),
                                (u'Ia3\u0305d',[hpkpl, [k,l], [twohpl4n,l],l4n]),
                                ('F23',[[hpk,hpl,kpl],[k,l],hpl,l]),
                                (u'Fm3\u0305',[[hpk,hpl,kpl],[k,l],hpl,l]),
                                ('F432',[[hpk,hpl,kpl],[k,l],hpl,l]),
                                (u'F4\u03053m',[[hpk,hpl,kpl],[k,l],hpl,l]),
                                (u'Fm3\u0305m',[[hpk,hpl,kpl],[k,l],hpl,l]),
                                ('F4₁32',[[hpk,hpl,kpl],[k,l],hpl,l4n]),
                                (u'F4\u03053c',[[hpk,hpl,kpl],[k,l],[h,l],l]),
                                (u'Fm3\u0305c',[[hpk,hpl,kpl],[k,l],[h,l],l]),
                                (u'Fd3\u0305',[[hpk,hpl,kpl],[kpl4n,k,l],hpl,l4n]),
                                (u'Fd3\u0305m',[[hpk,hpl,kpl],[kpl4n,k,l],hpl,l4n]),
                                (u'Fd3\u0305c',[[hpk,hpl,kpl],[kpl4n,k,l],[h,l],l4n])
                              ])

    def getcolumn(self,m):
        mh,mk,ml = m
        column = None
        for mh,mk,ml in [[mh,mk,ml],[mk,ml,mh],[ml,mh,mk]]:     # indicies are permutable for cubic (except #205)
            if arezeros(1,1,0,m):
                column = 3
            elif mh==mk:
                column = 2
            elif arezeros(1,0,0,m):
                column = 1

            if column:
                return column, [mh,mk,ml]
        column = 0
        return column, [mh,mk,ml]



Triclinic = Triclinic()
Monoclinic = Monoclinic()
Orthorhombic = Orthorhombic()
Tetragonal = Tetragonal()
Trigonal = Trigonal()
Hexagonal = Hexagonal()
Cubic = Cubic()


SGkeys = SGClass.conditions.keys()

crystalsystems = [Triclinic, Monoclinic, Orthorhombic,Tetragonal,Trigonal, Hexagonal, Cubic]


def check(m,SG):
    """
    This is the function intended for external use. Checks which crystal system the SG is in, then executes that
    system's check method.
    """

    # Check cache
    key=SG+','+''.join(map(str,m))
    if key in cache:
        return cache[key]

    for crystalsystem in crystalsystems:
        if SG in crystalsystem.conditions:
            reflected = crystalsystem.check(m,SG)
            cache[key]=reflected
            return reflected

if __name__=='__main__':
    def test(m,SG):
        print SG,m,check(m,SG)

    SG=u'P2₁'
    m=(1,2,1)
    test(m,SG)
    m=(0,1,0)
    test(m,SG)
    m=(0,2,0)
    test(m,SG)
    m=(0,3,0)
    test(m,SG)

    SG = u'P2₁2₁2'
    m=(1,1,1)
    test(m,SG)
    m=(0,1,0)
    test(m,SG)

    @debugtools.timeit
    def timetest():
        for i in range(0,300000):
            check(m,SG)
    timetest()

    m=(2,0,1)
    test(m,u'Pmn2₁')
