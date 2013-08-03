# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
# http://docs.python.org/2/reference/datamodel.html
from __future__ import division
import numpy as np

class IncompatibleUnits(Exception):
    def __init__(self, msg=None):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)
    
class Unit(np.ndarray):

    BASE_UNITS = ['m', 's', 'kg', 'K', 'mol', 'coul', 'candela']
    
    def __new__(cls, input_array, exponents=None, label=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.exponents = np.array(exponents)
        obj.label = label
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.exponents = getattr(obj, 'exponents', None)
        self.label = getattr(obj, 'label', None)
        
    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)
        
    def __str__(self):
        'pretty-print Unit object' 
        labelstrings = []
        for L,e in zip(Unit.BASE_UNITS, self.exponents):
            if e == 0.0:
                continue
            if np.abs(e % 1) < 1e-6:
                if e == 1.0:
                    labelstrings.append(L)
                else:
                    labelstrings.append('{0}^{1}'.format(L, int(e)))
            else:
                labelstrings.append('{0}^{1}'.format(L, e))
        LS = '*'.join(labelstrings)
        
        known_units = {'m^2*s^-2*kg':'J',
                       'm*s^-2*kg':'N',
                       'm^-1*s^-2*kg':'Pa'}
        LS = known_units.get(LS, LS)

        return '{0} * {1}'.format(np.array(self), LS)

    def __repr__(self):
        return 'Unit({0}, exponents={1}, label={2})'.format(np.array(self),
                                                            self.exponents,
                                                            self.label)
                                                            
    @property
    def value(self):
        return np.array(self)

    def as_units(self, U):
        '''print units in something other than the base units.
        >> print u.kJ
        1000.0 * J

        >> a = 5 * u.kJ
        >> print a
        5000 * J

        >> print a.AS(u.kJ)
        5 * u.kJ
        '''
        V = self / U
        if not isinstance(V, float):
            raise Exception('cannot cast those units')
        return '{0} * {1}'.format(float(V), U.label)

    as_ = AS = as_units

    def __format__(self, format_spec):
        fields = format_spec.split()
        u = units()
        if len(fields) == 1:
            f1 = '{{0:{0}}} * {{1}}'.format(fields[0])
            f2 = getattr(u, self.label)
            return f1.format(float(self/f2), self.label)
            
        elif len(fields) == 2:
            f1 = '{{0:{0}}} * {{1}}'.format(fields[0])
            
            f2 = getattr(u, fields[1])
            return f1.format(float(self/f2), self.label)
        else:
            return str(self)
        
        

    @property
    def latex(self):
        'return latex string of units'
        labelstrings = []
        for L,e in zip(Unit.BASE_UNITS, self.exponents):
            if e == 0.0:
                continue
            # check for near integer units
            if np.abs(e % 1) < 1e-6:
                if e == 1.0:
                    labelstrings.append(L)
                elif e > 0:
                    labelstrings.append('{0}^{1}'.format(L, int(e)))
                else:
                    labelstrings.append('{0}^{{{1}}}'.format(L, int(e)))
            else:
                labelstrings.append('{0}^{1}'.format(L, e))
        return '${0}$'.format(' '.join(labelstrings))
                
    ####################################################################
 
    def __lt__(self, other): raise Exception('Not Implemented')
    def __le__(self, other): raise Exception('Not Implemented')

    def __eq__(self, other):
        'implement a==b for units'
        if not isinstance(other, Unit):
            raise IncompatibleUnits(['you should not compare objects'
                                     ' with different units.'
                                     ' {0} == {1}'.format(self,other)])
        
        return (np.all(self.exponents == other.exponents)
                and np.all(np.array(self) == np.array(other)))

    def __ne__(self, other):
        return not self == other
    
    def __gt__(self, other): raise Exception('Not Implemented')
    def __ge__(self, other): raise Exception('Not Implemented')   

    ####################################################################  

    def __add__(self, other):
        if not hasattr(other, 'exponents'):
            raise IncompatibleUnits('addition')
        if not np.all(self.exponents == other.exponents):
            raise IncompatibleUnits('addition')
        return Unit(np.array(self) + np.array(other),
                    exponents=self.exponents,
                    label=self.label)

    def __radd__(self, other):
        return self + other
    
    def __iadd__(self, other):
        self = self + other
        return self
                
    ####################################################################
    
    def __sub__(self, other):
        if not hasattr(other, 'exponents'):
            raise IncompatibleUnits('subtraction, minus non unit')
        if not np.all(self.exponents == other.exponents):
            raise IncompatibleUnits('subtraction')
        return Unit(np.array(self) - np.array(other),
                    exponents=self.exponents,
                    label=self.label)
    
    def __rsub__(self, other):
        return self - other
    
    def __isub__(self, other):
        self = self - other
        return self

    ####################################################################

    def __mul__(self, other):
        'Unit * other'
        if isinstance(other, Unit):
            e1 = self.exponents
            e2 = other.exponents
                     
            if np.all((self.exponents + other.exponents) == 0.0):
                # dimensionless
                return np.array(self) * np.array(other)

            if ('*' in self. label
                or '/' in self.label
                or '^' in self.label):
                label1 = '({0})'.format(self.label)
            else:
                label1 = self.label

            if ('*' in other. label
                or '/' in other.label
                or '^' in other.label):
                label2 = '({0})'.format(other.label)
            else:
                label2 = other.label

            label = '{0}*{1}'.format(label1, label2)

            return Unit(np.array(self) * np.array(other),
                        exponents= e1 + e2,
                        label = label)
 
        else:
            # unit * non-unit
            return Unit(np.array(self) * other,
                    exponents=self.exponents,
                    label=self.label)
         
    def __rmul__(self, other):    
        return Unit(np.array(self) * other,
                    exponents=self.exponents,
                    label=self.label)
    
    def __imul__(self, other):
        self = self * other
        return self

    #####################################################################

    def __div__(self, other):
        if isinstance(other, Unit):
            e1 = self.exponents
            e2 = other.exponents
                     
            if np.all((self.exponents - other.exponents) == 0.0):
                # print 'div: result is dimensionless', self, other
                return np.array(self) / np.array(other)
           
            if ('*' in self. label
                or '/' in self.label
                or '^' in self.label):
                label1 = '({0})'.format(self.label)
            else:
                label1 = self.label

            if ('*' in other. label
                or '/' in other.label
                or '^' in other.label):
                label2 = '({0})'.format(other.label)
            else:
                label2 = other.label

            label = '{0}/{1}'.format(label1, label2)

            return Unit(np.array(self) / np.array(other),
                        exponents= e1 - e2,
                        label = label)
        else:
            # Unit / number
            U = Unit(np.array(self) / other,
                     exponents=self.exponents,
                     label=self.label)
            return U

    def __rdiv__(self, other):
        # this is for something like other / self
        value = other / np.array(self)
        return Unit(value, -self.exponents,
                    '{0}^-1'.format(self.label))
        
    
    def __idiv__(self, other):
        self = self / other
        return self

    def __floordiv__(self, other): raise Exception('Not Implemented')
    def __rfloordiv__(self, other): raise Exception('Not Implemented') 
    def __ifloordiv__(self, other): raise Exception('Not Implemented')    

    __truediv__ = __div__
    __rtruediv__ = __rdiv__    
    
    def __itruediv__(self, other):
        self = self / other
        return self
        
    #####################################################################
    
    def __pow__(self, other, modulo=None):
        return Unit(np.array(self)**other,
                    self.exponents * other,
                    self.label + '^{0}'.format(other))

    def __rpow__(self, other):
        raise IncompatibleUnits('You cannot raise a number to a unit!')
    
    def __ipow__(self, other, modulo=None):
        data = np.array(self)
        newexp = self.exponents * other
        newlabel = self.label + '^{0}'.format(other)
        self = Unit(data**other,
                    newexp, newlabel)
        return self

    #####################################################################
    
    def __mod__(self, other): raise Exception('Not Implemented')
    def __imod__(self, other): raise Exception('Not Implemented')
    def __rmod__(self, other): raise Exception('Not Implemented')
    def __divmod__(self): raise Exception('Not Implemented')
    def __rdivmod__(self): raise Exception('Not Implemented')

    #####################################################################

    def __lshift__(self, other): raise Exception('Not Implemented')
    def __rlshift__(self, other): raise Exception('Not Implemented')
    def __ilshift__(self, other): raise Exception('Not Implemented')
    def __rshift__(self, other): raise Exception('Not Implemented')
    def __rrshift__(self, other): raise Exception('Not Implemented')
    def __irshift__(self, other): raise Exception('Not Implemented')

    #####################################################################
    
    def __and__(self, other): raise Exception('Not Implemented')
    def __rand__(self, other): raise Exception('Not Implemented')
    def __iand__(self, other): raise Exception('Not Implemented')

    #####################################################################
    
    def __xor__(self, other): raise Exception('Not Implemented')
    def __rxor__(self, other): raise Exception('Not Implemented')
    def __ixor__(self, other): raise Exception('Not Implemented')

    #####################################################################
    
    def __or__(self, other): raise Exception('Not Implemented')
    def __ror__(self, other): raise Exception('Not Implemented')
    def __ior__(self, other): raise Exception('Not Implemented')

    #####################################################################

    def __neg__(self):
        return -1 * self
    def __pos__(self):
        return 1 * self
    
    def __abs__(self):
        return Unit(np.abs(np.array(self)), self.exponents, self.label)
    
    def __invert__(self): raise Exception('Not Implemented')

    #####################################################################
        
    def __complex__(self): raise Exception('Not Implemented')
    
    def __int__(self):
        return int(np.asarray(self, np.int))
                   
    def __long__(self):
        return long(np.asarray(self, np.long))
    
    def __float__(self):
        return float(np.asarray(self, np.float))

    #####################################################################

    def __oct__(self): raise Exception('Not Implemented')
    def __hex__(self): raise Exception('Not Implemented')

    def __index__(self): raise Exception('Not Implemented')
    
    def __coerce__(self): raise Exception('Not Implemented')

    #####################################################################
    def sin(self): raise Exception('Not Implemented')

    @staticmethod
    def base_units(input=None):
        '''define the base units. input is a string:
        SI
        MKS
        American
        atomic

        or a list of units, e.g.:
        ['m', 's', 'kg', 'K', 'mol', 'coul', 'candela']'''
        if input == 'SI' or input is None:
            BU = ['m', 's', 'kg', 'K', 'mol', 'coul', 'candela']
        elif input == 'MKS':
            BU = ['cm', 's', 'gm', 'K', 'mol', 'coul', 'candela']
        elif input == 'American':
            BU = ['in', 's', 'lb', 'R', 'mol', 'coul', 'candela']
        elif input == 'atomic':
            raise ['nm', 'fs', 'amu', 'K', 'molecule', 'coul', 'candela']
        else:
            BU = input

        Unit.BASE_UNITS = BU
        return BU

    @staticmethod
    def degC(C):
        '''Celcius to Kelvin
        >> u = units()
        >> T1 = u.degC(100)'''
        u = units()
        K = (C + 273.15) * u.K
        return K

    @staticmethod
    def degF(F):
        '''Fahrenheit to Kelvin
        >> u = cmu.units
        >> T1 = u.degF(100)'''
        C = Unit.degF2C(F)
        return  Unit.degC(C)

    @staticmethod
    def degF2C(F):
        '''Fahrenheit to Celcius
        no units attached to output!'''
        C = (F - 32) * 5 / 9
        return C

    @staticmethod
    def degC2F(C):
        '''Celsius to Fahrenheit
        no units attached to output!'''
        F = C * 9 / 5 + 32
        return F

    @staticmethod
    def degF2R(F):
        '''Fahrenheit to Rankine
        u = units()
        R = degF2R(212) # 212F in Rankine'''
        R = (F + 459.67) * u.R
        return R

    @staticmethod
    def degC2R(C):
        '''Celcius to Rankine
        >> u = units()
        >> R = u.degC2R(100)  # 100 degC in Rankine'''
        K = degC(C)
        R = 5 / 9 * float(K) * u.R
        return R



def units(base_unit_input=None):
    '''return a structure of units in the base_unit_input'''

    Unit.base_units(base_unit_input)

    LENGTH = {}
    LENGTH['km'] = 1000
    LENGTH['m'] = 1
    LENGTH['dm']=1e-1 
    LENGTH['cm'] = 1e-2
    LENGTH['mm'] = 1e-3
    LENGTH['um'] = 1e-6
    LENGTH['nm'] = 1e-9
    LENGTH['angstrom'] = 1e-10
    LENGTH['a0'] = 0.529e-10*LENGTH['m']
    LENGTH['Bohr'] = LENGTH['a0']
    LENGTH['in'] = 2.54*LENGTH['cm']
    LENGTH['mil'] = 1e-3*LENGTH['in']
    LENGTH['ft'] = 12*LENGTH['in']
    LENGTH['yd'] = 3*LENGTH['ft']
    LENGTH['mile'] = 5280*LENGTH['ft']
    LENGTH['furlong'] = 660*LENGTH['ft']
    LENGTH['chain'] = 66*LENGTH['ft']

    MASS = {}
    MASS['kg'] = 1e3
    MASS['gm'] = 1
    MASS['mg'] = 1e-3
    MASS['lb'] = 0.45359237 * MASS['kg']
    MASS['lbm'] = MASS['lb']
    MASS['oz'] = 1 / 16 * MASS['lb']
    MASS['amu'] = 1.660538782e-27 * MASS['kg']
    MASS['ton'] = 2000 * MASS['lb']
    MASS['tonne'] = 1000 * MASS['kg']
    MASS['longton'] = 2240 * MASS['lb']
    
    TIME = {}
    TIME['s'] = 1
    TIME['min'] = 60
    TIME['hr'] = 60 * TIME['min']
    TIME['day'] = 24 * TIME['hr']
    TIME['week'] = 7 * TIME['day']
    TIME['year'] = 365.242199 * TIME['day']
    
    TEMPERATURE = {}
    TEMPERATURE['K'] = 1
    TEMPERATURE['R'] = 5/9 * TEMPERATURE['K'] # I do not understand 
    # why this is 5/9. I
    # think it should be
    # 9/5, but then no
    # conversions work.
    TEMPERATURE['dC'] = TEMPERATURE['K'] # relative degree C
    TEMPERATURE['dF'] = TEMPERATURE['R'] # relative degree F
    
    MOL = {}
    # you cannot define lbmol or kgmol here because the masses may
    # not be normalized to the chosen base unit above. these units
    # are defined after the base-units are defined and all other
    # units are normalized. 
    MOL['mol'] = 1
    MOL['kmol'] = 1000
    MOL['mmol'] = 1e-3
    MOL['molecule'] = 6.022e-23
    
    CHARGE = {}
    CHARGE['coul'] = 1
    
    LUMINOSITY = {}
    LUMINOSITY['cd'] = 1
    
    ALL_UNITS = [LENGTH, TIME, MASS, TEMPERATURE, MOL, CHARGE, LUMINOSITY]

    # now we normalize all these defined units by the chosen base
    # units Users can define the base units in any order they
    # want. The strings for each UNIT is defined in BASE_UNITS.

    class U:
        'simple storage unit'
        pass
        

    UNITS = U()

    # here we loop through the units, and make them attributes of the
    # UNITS instance above
    for i,category in enumerate(ALL_UNITS):
        for baseunit in Unit.BASE_UNITS:
            if baseunit in category:
                NORMV = category[baseunit]
                
                for key,val in zip(category.keys(),
                               category.values()):
                    e = [0, 0, 0, 0, 0, 0, 0]
                    e[i] = 1
                    U = Unit(val/NORMV, e, key)
                    setattr(UNITS, key, U)

    # some derived mole units
    UNITS.lbmol = UNITS.lb / UNITS.gm  *  UNITS.mol
    UNITS.gmmol = UNITS.gm * UNITS.mol
    UNITS.kgmol = UNITS.kg / UNITS.gm * UNITS.mol
    UNITS.mmol = UNITS.mol / 1000
    UNITS.umol = UNITS.mol / 1e6

    #------- Volume -------
    UNITS.cc = UNITS.cm**3           
    UNITS.L = 1000 * UNITS.cc           
    UNITS.mL = UNITS.cc               
    UNITS.floz = 29.5735297 * UNITS.cc  
    UNITS.pint = 473.176475 * UNITS.cc  
    UNITS.quart = 946.35295 * UNITS.cc  
    UNITS.gal = 3.78541197 * UNITS.L    

    #---- frequency ----
    UNITS.Hz = 1 / UNITS.s       
    UNITS.kHz = 1e3  * UNITS.Hz  
    UNITS.MHz = 1e6  * UNITS.Hz
    UNITS.GHz = 1e9  * UNITS.Hz

    #---- force -------
    UNITS.N = UNITS.kg * UNITS.m / UNITS.s**2   
    UNITS.dyne = 1e-5 * UNITS.N      
    UNITS.lbf = 4.44822 * UNITS.N    

    #----- energy -----
    UNITS.J = UNITS.kg * UNITS.m**2 / UNITS.s**2 
    UNITS.MJ = 1e6 * UNITS.J         
    UNITS.kJ = 1e3 * UNITS.J         
    UNITS.mJ = 1e-3 * UNITS.J        
    UNITS.uJ = 1e-6 * UNITS.J        
    UNITS.nJ = 1e-9 * UNITS.J        
    UNITS.eV = 1.6022e-19 * UNITS.J    
    UNITS.BTU = 1.0550559e3 * UNITS.J  
    UNITS.kWh = 3.6e6 * UNITS.J        
    UNITS.cal = 4.1868 * UNITS.J       
    UNITS.kcal = 1e3 * UNITS.cal       
    UNITS.erg = 1e-7 * UNITS.J

    #---- pressure -----
    UNITS.Pa = UNITS.N / UNITS.m**2
    UNITS.kPa = 1000 * UNITS.Pa
    UNITS.MPa = 1e6 * UNITS.Pa
    UNITS.GPa = 1e9 * UNITS.Pa
    UNITS.torr = 133.322 * UNITS.Pa
    UNITS.mtorr = 1e-3 * UNITS.torr
    UNITS.bar = 1e5 * UNITS.Pa
    UNITS.mbar = 1e-3 * UNITS.bar
    UNITS.atm = 1.013e5 * UNITS.Pa
    UNITS.psi = 6.895e3 * UNITS.Pa
    UNITS.mmHg = 1 / 760 * UNITS.atm

    #----- power --- ---
    UNITS.W = UNITS.J / UNITS.s
    UNITS.MW = 1e6 * UNITS.W
    UNITS.kW = 1e3 * UNITS.W
    UNITS.mW = 1e-3 * UNITS.W
    UNITS.uW = 1e-6 * UNITS.W
    UNITS.nW = 1e-9 * UNITS.W
    UNITS.pW = 1e-12 * UNITS.W
    UNITS.hp = 745.69987 * UNITS.W

    #------ Voltage -----
    UNITS.V = UNITS.J / UNITS.coul
    UNITS.kV = 1e3 * UNITS.V
    UNITS.mV = 1e-3 * UNITS.V
    UNITS.uV = 1e-6 * UNITS.V

    #----- Current ------
    UNITS.A = UNITS.coul / UNITS.s
    UNITS.mA = 1e-3 * UNITS.A
    UNITS.uA = 1e-6 * UNITS.A
    UNITS.nA = 1e-9 * UNITS.A

    #----magnetic field -----
    UNITS.T = UNITS.V * UNITS.s / UNITS.m**2
    UNITS.tesla = UNITS.T

    UNITS.gauss = 1e-4 * UNITS.T

    #----area----------------
    UNITS.acre = 4840 * UNITS.yd**2
    UNITS.hectare = 10000 * UNITS.m**2

    #----electromagnetic units-----
    UNITS.ohm = UNITS.V / UNITS.A
    UNITS.H = UNITS.ohm * UNITS.s # Henry
    UNITS.Wb = UNITS.V * UNITS.s # Weber
    UNITS.S = 1 / UNITS.ohm # siemens
    UNITS.siemens = UNITS.S

    UNITS.F = UNITS.coul / UNITS.V # farad
    UNITS.farad = UNITS.F

    UNITS.X = UNITS.dimensionless = Unit(1, [0, 0, 0, 0, 0, 0, 0], 'dimensionless')
    
    for attr in dir(UNITS):
        if isinstance(getattr(UNITS, attr), Unit):
            u = getattr(UNITS, attr)
            setattr(u, 'label', attr)

    UNITS.degC = Unit.degC
    UNITS.degF = Unit.degF
    UNITS.degF2C = Unit.degF2C
    UNITS.degC2F = Unit.degC2F
    UNITS.degC2R = Unit.degC2R
    UNITS.degF2R = Unit.degF2R

    return UNITS

if __name__ == '__main__':
    u = units('SI')
    Eins = 55 * u.kJ/u.mol
    R = 8.314 * u.J/u.mol/u.K

    T = (np.linspace(800, 1000) + 273.15) * u.K

    Cso = 8.3 * np.exp(-Eins/ R / T)

    import matplotlib.pyplot as plt
    plt.plot(T, Cso)
    plt.xlabel('Temperature(K)')
    plt.ylabel('CsO (at. %)')
    plt.show()
