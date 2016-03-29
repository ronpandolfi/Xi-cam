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
        col=self.getcolumn(m)
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
                if not condition(*m):
                    return False
        return True

class Triclinic(SGClass):
    conditions = OrderedDict([('P1',[]),
                              ('P-1',[])])
    def check(self,m,SG):
        return True

class Monoclinic(SGClass):
    ############## IMPORTANT NOTE ################
    # Monoclinic must have the unique axis as b for this to work!

    conditions = OrderedDict([('P2',[None,None,None]),
                              (u'P2₁',[None,None,k]),
                              ('C2',[hpk,h,k]),
                              ('Pm',[None,None,None]),
                              ('Pc',[None,l,None]),
                              ('Cm',[hpk,h,k]),
                              ('Cc',[hpk,[h,l],k]),
                              ('P2/m',[None,None,None]),
                              (u'P2₁/m',[None,None,k]),
                              ('C2/m',[hpk,h,k]),
                              ('P2/c',[None,l,None]),
                              (u'P2₁/c',[None,l,k]),
                              ('C2/c',[hpk,[h,l],k])])



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
        return column

class Orthorhombic(SGClass):
    conditions = OrderedDict([('P222',[None,None,None,None,None,None,None]), #16
                              (u'P222₁',[None,None,None,None,None,None,l]),
                              (u'P2₁2₁2',[None,None,None,None,h,k,None]),
                              (u'P2₁2₁2₁',[None,None,None,None,h,k,l]),
                              (u'C222₁',[hpk,k,h,hpk,h,k,l]),
                              ('C222',[hpk,k,h,hpk,h,k,None]),
                              ('F222',[[hpk,hpl,kpl],[k,l],[h,l],[h,k],h,k,l]),
                              ('I222',[hpkpl,kpl,hpl,hpk,h,k,l]),
                              (u'I2₁2₁2₁',[hpkpl,kpl,hpl,hpk,h,k,l]),
                              ('Pmm2',[None,None,None,None,None,None,None]),
                              (u'Pmc2₁',[None,None,l,None,None,None,l]),
                              ('Pcc2',[None,l,l,None,None,None,l]),
                              ('Pma2',[None,None,h,None,h,None,None]),
                              (u'Pca2₁',[None,l,h,None,h,None,l]),
                              ('Pnc2',[None,kpl,l,None,None,k,l]),
                              (u'Pmn2₁',[None,None,hpl,None,h,None,l])]) #31




                              #     u'Pmn2₁', 'Pba2', u'Pna2₁', 'Pnn2', 'Cmm2', u'Cmc2₁',
                              # 'Ccc2', 'Amm2',
                              # 'Aem2', 'Ama2', 'Aea2', 'Fmm2', 'Fdd2', 'Imm2', 'Iba2', 'Ima2', 'Pmmm', 'Pnnn', 'Pccm',
                              # 'Pban', 'Pmma',
                              # 'Pnna', 'Pmna', 'Pcca', 'Pbam', 'Pccn', 'Pbcm', 'Pnnm', 'Pmmn', 'Pbcn', 'Pbca', 'Pnma',
                              # 'Cmcm', 'Cmce',
                              # 'Cmmm', 'Cccm', 'Cmme', 'Ccce', 'Fmmm', 'Fddd', 'Immm', 'Ibam', 'Ibca', 'Imma']

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
        return column

class Tetragonal(SGClass):
    conditions = OrderedDict([])

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
        return column

class Trigonal(SGClass):
    conditions = OrderedDict([])

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
                column = 6
            elif mh==mk:
                column = 5
            else:
                column = 4
        else:
            debugtools.frustration()
            raise ValueError
        return column

class Hexagonal(SGClass):
    conditions = OrderedDict([])

    def getcolumn(self,m):
        mh,mk,mi,ml = m
        if arezeros(1,1,1,0,m):
            column = 2
        elif mh==mk and mi==-2*mh:          # TODO: What rules apply to hexagonal 1111
            column = 1
        elif arezeros(0,0,1,0,m) and mh==-mk:
            column = 0
        else:
            column = None
        return column

class Cubic(SGClass):
    conditions = OrderedDict([])

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
                return column
        column = 0
        return column



Triclinic = Triclinic()
Monoclinic = Monoclinic()
Orthorhombic = Orthorhombic()

SGkeys = SGClass.conditions.keys()

crystalsystems = [Triclinic, Monoclinic, Orthorhombic]


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
