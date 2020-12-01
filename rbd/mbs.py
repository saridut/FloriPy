#!/usr/bin/env python

import sys
import weakref

import numpy as np
import numpy.linalg as la

import networkx as nx
import pygraphviz
import h5py

from floripy.file_formats import yamlio
from floripy.rbd import bodymodels
from floripy.rbd import jointmodels
from floripy.mathutils import xform as tr


class Mbs(object):
    def __init__(self, **kwargs):
        fn_in = kwargs['fn_model']
        fn_out = kwargs['fn_out']
        restart = kwargs['restart']

        self.__draw_graph = kwargs.pop('draw_graph', False)
        self.__rooted = kwargs.pop('rooted', False)
        self.__pullback = kwargs.pop('pullback', False)
        if self.__pullback:
            self.__pullback_radius = kwargs['pullback_radius']
            self.__pullback_r = np.zeros((3,))

        if fn_in is not None:
            config = yamlio.read(fn_in)
            self.create_mbs(config['bodies'], config['joints'])

        if restart:
            self.__fh_out = h5py.File(fn_out,'a')
            if 'ts' in self.__fh_out:
                self.__h5cntr = len(self.__fh_out['ts'])
            else:
                self.__h5cntr = 0
        else:
            self.__fh_out = h5py.File(fn_out,'w')
            self.__h5cntr = 0
            self.tofile(kwargs['time_start'])




    def create_mbs(self, body_dict=None, joint_dict=None):
        self.__bodies = {0: bodymodels.create(0, 'zero')}
        self.__joints = {}
        self.__msa = None         #Minimum spanning arborescence
        self.__predecessors = {0:None}
        self.__successors = {0:[]}
        self.__ancestors = {0:[]}
        self.__descendants = {0:[]}

        #Add all bodies
        if body_dict is not None:
            for key, val in body_dict.items():
                geom, geom_par = val.popitem() #geom_par is a dict
                abody = bodymodels.create(key, geom, **geom_par)
                self.__bodies[key] = abody

            self.__num_bodies = len(self.__bodies) - 1
            for k in self.bid_iter():
                self.__successors[k] = []
                self.__ancestors[k] = []
                self.__descendants[k] = []

        #Add all joints
        if joint_dict is not None:
            #Add pf and sf for root joint. Also ensure that there is exactly
            #one root joint.
            nrj = 0
            for joint in joint_dict.values():
                if joint['joint_type'] == 'root':
                    joint['pf'] = 0
                    joint['sf'] = 1
                    nrj += 1
#           assert nrj == 1

            #Construct the model digraph
            model_digraph = nx.DiGraph()
            for key, val in joint_dict.items():
                model_digraph.add_edge(val['pf'], val['sf'], {'key_in_joint_dict':key})
            self.__num_joints = model_digraph.number_of_edges()

            #Construct the minimum spanning arborescence
            if nx.is_arborescence(model_digraph):
                self.__msa = model_digraph.copy()
            else:
                self.__msa = nx.minimum_spanning_arborescence(model_digraph)
            self.__num_tree_joints = self.__msa.number_of_edges()
            if self.__draw_graph :
                self.export_digraph(model_digraph, 'model_digraph.eps') 
                self.export_digraph(self.__msa, 'msa.eps') 

            for edge in self.__msa.edges_iter():
                u, v = edge
                #Copy the joint attributes from model_digraph to self.__msa
                self.__msa[u][v]['key_in_joint_dict'] = model_digraph[u][v]['key_in_joint_dict']
                self.__predecessors[v] = u
                self.__successors[u].append(v)

            #Fill in the ancestors and descendants lists
            for k in self.bid_iter():
                predecessor = self.__predecessors[k]
                self.__ancestors[k].append(predecessor)
                self.__ancestors[k].extend(self.__ancestors[predecessor])
                while predecessor is not None:
                    self.__descendants[predecessor].append(k)
                    predecessor = self.__predecessors[predecessor]

            for k in self.bid_iter():
                pk = self.__predecessors[k]
                key_in_joint_dict = self.__msa[pk][k]['key_in_joint_dict']
                joint_par = joint_dict[key_in_joint_dict]
                joint_type = joint_par.pop('joint_type')
                joint_par['category'] = 'tree'
                joint_par['pf'] = weakref.proxy(self.__bodies[pk])
                joint_par['sf'] = weakref.proxy(self.__bodies[k])
                self.__joints[k] = jointmodels.create(joint_type, **joint_par)

            #Remove the msa edges from model_digraph
            model_digraph.remove_edges_from(self.__msa.edges())
            self.__num_loop_joints = model_digraph.number_of_edges()

            #Create the loop joints after renumbering
            ljid = self.__num_tree_joints
            for edge in model_digraph.edges_iter():
                ljid += 1
                u, v = edge
                key_in_joint_dict = model_digraph[u][v]['key_in_joint_dict']
                joint_par = joint_dict[key_in_joint_dict]
                joint_type = joint_par.pop('joint_type')
                joint_par['category'] = 'loop'
                joint_par['pf'] = weakref.proxy(self.__bodies[u])
                joint_par['sf'] = weakref.proxy(self.__bodies[v])
                self.__joints[ljid] = jointmodels.create(joint_type, **joint_par)

        assert len(self.__bodies) == self.__num_tree_joints + 1

        #Loop over the tree joints to find the total number of generalized
        #coordinates and generalized speeds.
        self.__num_gencoord = 0
        self.__num_genspeed = 0
        self.__gsloc = {}
        for jid in self.tree_jid_iter():
            joint = self.__joints[jid]
            ngc = joint.num_gencoord
            ngs = joint.num_genspeed
            if ngs != 0:
                self.__gsloc[jid] = (self.__num_genspeed,
                                        self.__num_genspeed+ngs)
                self.__num_gencoord += ngc
                self.__num_genspeed += ngs
        self.__gencoord = np.zeros((self.__num_gencoord,))
        self.__gencoord_dot = np.zeros((self.__num_gencoord,))
        self.__genspeed = np.zeros((self.__num_genspeed,))

        #Allocate space for the partial matrix
        self.__partial_mat = np.zeros((6*self.__num_bodies,
                                            self.__num_genspeed))

        #Loop over the loop joints to find the total number of constraints.
        self.__num_constraints = 0
        self.__cloc = {}
        for jid in self.loop_jid_iter():
            joint = self.__joints[jid]
            nc = joint.num_constraints
            self.__cloc[jid] = (self.__num_constraints,
                                    self.__num_constraints+nc)
            self.__num_constraints += nc

        #Allocate space for the constraint matrix
        self.__constraint_mat = np.zeros((self.__num_constraints,
                                            self.__num_genspeed))

        #Allocate space for the evaluated constraints
        self.__plc = np.zeros((self.__num_constraints,)) #Position level
        self.__vlc = np.zeros((self.__num_constraints,)) #Velocity level
#       Print statements for debugging
#       print(self.__predecessors)
#       print(self.__successors)
#       print(self.__ancestors)
#       print(self.__descendants)
#       for jid in self.jid_iter():
#           joint = self.__joints[jid]
#           print(jid, joint.connects)
#       print('num_constraints', self.__num_constraints)
#
#       Root joint state
        if self.__rooted:
            self.__pullback = False
        else:
            self.__joints[1].uproot()


    def export_digraph(self, G, fn):
        '''Draws a digraph to a file.

        '''
        if 'pygraphviz' in sys.modules:
            agraph = nx.to_agraph(G)
            agraph.draw(fn, format='eps', prog='neato')


    def get_torque_force():
        pass


    def update(self, q, time):
        #Update the joints
        i = 0
        for jid in self.tree_jid_iter():
            joint = self.__joints[jid]
            num_gencoord = joint.num_gencoord
            joint.update(q[i:i+num_gencoord])
            i += num_gencoord
        #Update the partial velocity matrix
        for jid in self.tree_jid_iter():
            joint = self.__joints[jid]
            jpvm = joint.get_jpvm()
            mb = 6*(jid-1)
            me = mb + 6
            nb, ne = self.__gsloc[jid]
            self.__partial_mat[mb:me,nb:ne] = jpvm
            shifter66 = self.__joints[jid].get_shifter66()
            p, s = joint.connects
            for j in self.__ancestors[s]:
                if j == 0:
                    break
                lb = 6*(p-1)
                le = lb + 6
                nb, ne = self.__gsloc[j]
                ppvm = self.__partial_mat[lb:le, nb:ne]
                self.__partial_mat[mb:me, nb:ne] = np.dot(shifter66, ppvm)
        #Update the constraint matrix
        self._update_constraint()


    def _update_constraint(self):
        '''Updates the constraint matrix

        '''
        if self.__num_constraints == 0:
            return
        #Update the constraint matrix 
        self.__constraint_mat[:,:] = 0.0
        for jid in self.loop_jid_iter():
            joint = self.__joints[jid]
            p, s = joint.connects
            pfcm, sfcm = joint.get_constraint_mat()
            mb, me = self.__cloc[jid]
            self.__constraint_mat[mb:me,:] = (
                            np.dot(sfcm, self.__partial_mat[6*s-6:6*s,:])
                            - np.dot(pfcm, self.__partial_mat[6*p-6:6*p,:]))


    def eval_position_constraints(self):
        for s in self.shadow_bid_iter():
            sb = self.__bodies[s]
            ob = self.__bodies[sb.bid_original]
            shifter = sb.get_shifter() - ob.get_shifter()
            m = s - (self.__num_original_bodies + 1)
            self.__plc[6*m] = shifter[2,1]
            self.__plc[6*m+1] = shifter[2,0]
            self.__plc[6*m+2] = shifter[1,0]
            self.__plc[6*m+3:6*m+6] = sb.get_com() - ob.get_com()
        return self.__plc


    def eval_velocity_constraints(self):
        self.__vlc = np.dot(self.__constraint_mat, self.__genspeed)
        return self.__vlc


    def constraints_violated(self, kind, tol):
        if kind == 'p':
            violation = self.eval_position_constraints()
        elif kind == 'v':
            violation = self.eval_velocity_constraints()
        else:
            raise ValueError(
                'Unknown value {0} for argument <kind>'.format(kind))
        if la.norm(violation, np.inf) > tol:
            violated = True
        else:
            violated = False
        return violated

    
    def gencoord_correction(self):
        pass


    def partition_gencoord(self):
        B = np.copy(self.__constraint_mat)
        print('rank = ', la.matrix_rank(B))
        m, n = B.shape
        assert m <= n
        rperm = np.empty((m,), dtype=np.int32)
        cperm = np.empty((m,), dtype=np.int32)
        rank = 0
        for k in range(m):
            nr, nc = divmod(np.argmax(abs(B[k:,k:])), n-k)
            mu = k + nr
            lamda = k + nc
            rperm[k] = mu
            cperm[k] = lamda
            #Row swap
            tmp = np.copy(B[k,k:])
            B[k,k:] = B[mu,k:]
            B[mu,k:] = tmp
            #Column swap
            tmp = np.copy(B[:,k])
            B[:,k] = B[:,lamda]
            B[:,lamda] = tmp
            #Forward elimination
            if abs(B[k,k]) >= np.finfo(np.float64).eps:
                rank += 1
                B[k+1:,k] /= B[k,k]
                B[k+1:, k+1:] -= np.outer(B[k+1:,k], B[k,k+1:])
            else:
                print('B----------------')
                print(B)
#       print('BD-------------')
#       BD = np.copy(B[0:rank, 0:rank])
#       for i in range(rank):
#           for j in range(i):
#               BD[i,j] = 0.0
#       print(BD)
#       print('BI-------------')
#       BI = np.copy(B[0:rank, rank:])
#       print(BI)
#       print('BD_inv_BI-------------')
#       print(np.dot(la.inv(BD), BI))
        #Backward substitution
        for k in range(rank-1,-1,-1):
            B[k,rank:] = (B[k,rank:] - np.dot(B[k,k+1:rank], B[k+1:rank,rank:]))/B[k,k]
        BD_inv_BI = B[0:rank,rank:]
#       print('BD_inv_BI numerical---')
#       print(BD_inv_BI)
#       print('BI numerical---')
#       print(np.dot(BD, BD_inv_BI))
#       raise SystemExit()

        return BD_inv_BI, cperm, rank


    def pullback(self):
        if self.__pullback:
            root_joint = self.__joints[1]
            r = root_joint.reroot(root_sphere_radius=self.__pullback_radius)
            self.__pullback_r += r
            pulled_back = True
            for jid in self.tree_jid_iter():
                joint = self.__joints[jid]
                joint.selfupdate()
        else:
            pulled_back = False
        return pulled_back


    def renormalize(self):
        '''Renormalizes unit quaternions in the joints.

        '''
        for jid in self.tree_jid_iter():
            joint = self.__joints[jid]
            joint.renormalize()
            joint.selfupdate()


    def get_gencoord(self):
        '''
        Called by timestepper to set initial condition.
        '''
        i = 0
        for jid in self.tree_jid_iter():
            joint = self.__joints[jid]
            num_gencoord = joint.num_gencoord
            self.__gencoord[i:i+num_gencoord] = joint.get_gencoord()
            i += num_gencoord
        return self.__gencoord


    def get_gencoord_dot(self, u):
        #Update velocity and angular velocity of the bodies
        velocities = np.dot(self.__partial_mat, u)
        for k in self.bid_iter():
            self.__bodies[k].update_velocity(velocities[6*k-6:6*k])
        i = 0
        j = 0
        for jid in self.tree_jid_iter():
            joint = self.__joints[jid]
            num_gencoord = joint.num_gencoord
            num_genspeed = joint.num_genspeed
            self.__genspeed[j:j+num_genspeed] = u[j:j+num_genspeed]
            self.__gencoord_dot[i:i+num_gencoord] = joint.get_gencoord_dot(
                                                        u[j:j+num_genspeed])
            i += num_gencoord
            j += num_genspeed
        return self.__gencoord_dot


    def get_partial_mat(self):
        return self.__partial_mat


    def get_constraint_mat(self):
        return self.__constraint_mat


    def get_coms(self):
        '''
        Return com for all bodies.
        '''
        coms = []
        for i in self.bid_iter():
            coms.append(self.__bodies[i].get_com())
        return coms


    def get_shifters(self):
        '''
        Return shifters for all bodies.
        '''
        shifters = []
        for i in self.bid_iter():
            shifters.append(self.__bodies[i].get_shifter())
        return shifters


    def get_all_radius(self):
        all_radius = []
        for i in self.bid_iter():
            body = self.__bodies[i]
            all_radius.append(body.get_radius())
        return all_radius


    def get_all_prolate_a(self):
        all_prolate_a = []
        for i in self.bid_iter():
            body = self.__bodies[i]
            all_prolate_a.append(body.get_prolate_a())
        return all_prolate_a


    def get_all_prolate_c(self):
        all_prolate_c = []
        for i in self.bid_iter():
            body = self.__bodies[i]
            all_prolate_c.append(body.get_prolate_c())
        return all_prolate_c


    def get_all_oblate_a(self):
        all_oblate_a = []
        for i in self.bid_iter():
            body = self.__bodies[i]
            all_oblate_a.append(body.get_oblate_a())
        return all_oblate_a


    def get_all_oblate_c(self):
        all_oblate_c = []
        for i in self.bid_iter():
            body = self.__bodies[i]
            all_oblate_c.append(body.get_oblate_c())
        return all_oblate_c


    def get_all_ellipsoid_a(self):
        all_ellipsoid_a = []
        for i in self.bid_iter():
            body = self.__bodies[i]
            all_ellipsoid_a.append(body.get_ellipsoid_a())
        return all_ellipsoid_a


    def get_all_ellipsoid_b(self):
        all_ellipsoid_b = []
        for i in self.bid_iter():
            body = self.__bodies[i]
            all_ellipsoid_b.append(body.get_ellipsoid_b())
        return all_ellipsoid_b


    def get_all_ellipsoid_c(self):
        all_ellipsoid_c = []
        for i in self.bid_iter():
            body = self.__bodies[i]
            all_ellipsoid_c.append(body.get_ellipsoid_c())
        return all_ellipsoid_c


    def get_all_director(self):
        '''
        Return director vector for all bodies.
        '''
        directors = []
        for i in self.bid_iter():
            directors.append(self.__bodies[i].get_director())
        return directors


    def get_all_ellipsoid_shifters(self):
        '''
        Return the ellipsoid shifters for all bodies.
        '''
        shifters = []
        for i in self.bid_iter():
            shifters.append(self.__bodies[i].get_ellipsoid_shifter())
        return shifters


    def jid_iter(self):
        '''Returns a range over the joint ids for looping sequentially.

        '''
        return range(1, self.__num_joints+1)


    def tree_jid_iter(self):
        '''Returns a range over the tree joint ids for looping sequentially.

        '''
        return range(1, self.__num_tree_joints+1)


    def loop_jid_iter(self):
        '''Returns a range over the loop joint ids for looping sequentially.

        '''
        return range(self.__num_tree_joints+1, self.__num_joints+1)


    def bid_iter(self):
        '''Returns a range over the body ids. The zero body is not
        included.

        '''
        return range(1, self.__num_bodies+1)


    def tofile(self, time):
        '''
        Write to output file.
        '''
        body_coms = np.zeros((self.num_bodies,3))
        body_oris = np.zeros((self.num_bodies,4))
        body_ang_vels = np.zeros((self.num_bodies,3))
        body_lin_vels = np.zeros((self.num_bodies,3))

        for bid in self.bid_iter():
            body = self.__bodies[bid]
            body_coms[bid-1,:] = self.__pullback_r + body.get_com()
            shifter = body.get_shifter()
            body_oris[bid-1,:] = tr.dcm_to_quat(shifter.T)
            velocity = body.get_velocity()
            body_ang_vels[bid-1,:] = velocity[0:3]
            body_lin_vels[bid-1,:] = velocity[3:6]

        #Write out time, body_coms, body_oris, body_ang_vels, and body_lin_vels
        grp_name = 'ts/{0}'.format(self.__h5cntr)
        self.__h5cntr += 1
        grp = self.__fh_out.create_group(grp_name)
        grp.attrs['time'] = time
        grp.create_dataset('body_coms', data=body_coms)
        grp.create_dataset('body_orientations', data=body_oris)
        grp.create_dataset('body_ang_vels', data=body_ang_vels)
        grp.create_dataset('body_lin_vels', data=body_lin_vels)


    def close_files(self):
        self.__fh_out.close()


    @property
    def num_bodies(self):
        return self.__num_bodies


    @property
    def num_joints(self):
        return self.__num_joints


    @property
    def num_gencoord(self):
        return self.__num_gencoord


    @property
    def num_genspeed(self):
        return self.__num_genspeed


    @property
    def num_constraints(self):
        return self.__num_constraints
