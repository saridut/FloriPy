#!/usr/bin/env python
'''
Class implementing a body.
'''
import math
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from floripy.mathutils import geometry as geom


class BodyBase:
    __metaclass__ = ABCMeta
    def __init__(self, bid):
        self._bid = bid
        self._com = np.zeros((3,))
        self._shifter = np.eye(3)
        self._velocity = np.zeros((6,))
        self._acceleration = np.zeros((6,))


    def update_bid(self, bid):
        '''Update the body id.

        '''
        self._bid = bid


    def update_position(self, com, shifter):
        '''
        Update com and shifter.
        '''
        self._com[:] = com
        self._shifter[:,:] = shifter


    def update_velocity(self, v):
        '''
        Update velocity and angular velocity
        '''
        self._velocity[:] = v


    def update_acceleration(self, a):
        '''
        Update velocity and angular acceleration
        '''
        self._acceleration[:] = a


    def update_com(self, com):
        self._com[:] = com


    def update_shifter(self, shifter):
        self._shifter[:,:] = shifter


    def get_com(self):
        return self._com


    def get_shifter(self):
        return self._shifter


    def get_velocity(self):
        return self._velocity


    def get_angular_velocity(self):
        return self._velocity[0:3]


    def get_linear_velocity(self):
        return self._velocity[3:6]


    def get_acceleration(self):
        return self._acceleration


    def get_angular_acceleration(self):
        return self._acceleration[0:3]


    def get_linear_acceleration(self):
        return self._acceleration[3:6]


    @abstractmethod
    def __repr__(self):
        pass


    @property
    def bid(self):
        return self._bid


class BodyZero(BodyBase):
    def __init__(self, bid=None):
        super(BodyZero, self).__init__(0)


    def __repr__(self):
        return 'zero'


    def update_position(self, com, shifter):
        raise RuntimeError('bodyzero cannot be updated')


class Sphere(BodyBase):
    def __init__(self, bid, radius=None):
        assert bid != 0
        super(Sphere, self).__init__(bid)
        if radius is None:
            raise ValueError('radius cannot be None.')
        self._radius = radius


    def __repr__(self):
        return 'sphere: radius = {0}\n'.format(self._radius)


    def get_radius(self):
        return self._radius


class Prolate(BodyBase):
    def __init__(self, bid, a=None, c=None):
        assert bid != 0
        super(Prolate, self).__init__(bid)
        if a is None:
            raise ValueError('a cannot be None.')
        if c is None:
            raise ValueError('c cannot be None.')
        #For an prolate a /= b = c. The director is along the +ve x-axis. This
        #is the reference configuration as per Kim and Karila. It is important to
        #keep this reference configuration as the hydrodynamic expressions depend
        #on this.
        self._a = a
        self._c = c
        self._d = np.array([1.0, 0.0, 0.0]) #Director

    def get_prolate_a(self):
        return self._a


    def get_prolate_c(self):
        return self._c


    def get_director(self):
        return np.dot(self._shifter, self._d)


    def __repr__(self):
        return ('oblate: a = b = {0}, c = {1}'.format(self._a, self._c)
                + 'director: '
                + ', '.join([str(x) for x in self._d])
                +'\n')


class Oblate(BodyBase):
    def __init__(self, bid, a=None, c=None):
        super(Oblate, self).__init__(bid)
        if a is None:
            raise ValueError('a cannot be None.')
        if c is None:
            raise ValueError('c cannot be None.')
        #For an oblate a = b /= c. The director is along the +ve z-axis. This is
        #the reference configuration as per Kim and Karila. It is important to
        #keep this reference configuration as the hydrodynamic expressions depend
        #on this.
        self._a = a
        self._c = c
        self._d = np.array([0.0, 0.0, 1.0]) #Director


    def get_oblate_a(self):
        return self._a


    def get_oblate_c(self):
        return self._c


    def get_director(self):
        return np.dot(self._shifter, self._d)


    def __repr__(self):
        return ('oblate: a = {0}, b = c = {1}'.format(self._a, self._c)
                + 'director: '
                + ', '.join([str(x) for x in self._d])
                +'\n')


class Parallelogram(BodyBase):
    def __init__(self, bid, vertices=None):
        if vertices is None:
            raise ValueError('vertices cannot be None.')
        super(Parallelogram, self).__init__(bid)
        self._vertices = np.asarray(vertices, dtype=np.float64)
        self._num_vertices = 4
        assert self._num_vertices == self._vertices.shape[0]
        a, c, dcm = geom.fit_ellipse(np.copy(self._vertices))
        self._a = a
        self._b = 'zero'
        self._c = c
        self._es = dcm.T #ellipsoid shifter


    def get_ellipsoid_a(self):
        return self._a


    def get_ellipsoid_b(self):
        return self._b


    def get_ellipsoid_c(self):
        return self._c


    def get_ellipsoid_shifter(self):
        es = np.dot(self._shifter, self._es)
        return es


    def __repr__(self):
        return ('parallelogram')


    @property
    def num_vertices(self):
        return self._num_vertices


def create(bid, body_type, **kwargs):
    classes = {'zero': BodyZero, 'sphere': Sphere, 'prolate': Prolate,
            'oblate': Oblate, 'parallelogram': Parallelogram}
    if body_type == 'zero':
        bid = 0
    return classes[body_type](bid, **kwargs)
