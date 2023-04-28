# Taken from pbdlib-python
import numpy as np

def periodic_clip(val, n_min, n_max):
        ''' keeps val within the range [n_min, n_max) by assuming that val is a periodic value'''
        if val < n_max and val >= n_min:
                val = val
        elif val >= n_max:
                val = val - (n_max - n_min)
        elif val < n_max:
                val = val + (n_max - n_min)

        return val


def tri_elipsoid(n_rings, n_points):
        ''' Compute the set of triangles that covers a full elipsoid of n_rings with n_points per ring'''
        tri = []
        for n in range(n_points - 1):
                # Triange down
                #       *    ring i+1
                #     / |
                #    *--*    ring i
                tri_up = np.array([n, periodic_clip(n + 1, 0, n_points),
                                                   periodic_clip(n + n_points + 1, 0, 2 * n_points)])
                # Triangle up
                #    *--*      ring i+1
                #    | /
                #    *    ring i

                tri_down = np.array([n, periodic_clip(n + n_points + 1, 0, 2 * n_points),
                                                         periodic_clip(n + n_points, 0, 2 * n_points)])

                tri.append(tri_up)
                tri.append(tri_down)

        tri = np.array(tri)
        trigrid = tri
        for i in range(1, n_rings - 1):
                trigrid = np.vstack((trigrid, tri + n_points * i))

        return np.array(trigrid)

def plot_gauss3d(ax, mean, covar, n_points=30, n_rings=20, color='red', alpha=0.3,
                                 linewidth=0):
        ''' Plot 3d Gaussian'''

        # Compute eigen components:
        (D0, V0) = np.linalg.eig(covar)
        U0 = np.real(V0.dot(np.diag(D0) ** 0.5))

        # Compute first rotational path
        psi = np.linspace(0, np.pi * 2, n_rings, endpoint=True)
        ringpts = np.vstack((np.zeros((1, len(psi))), np.cos(psi), np.sin(psi)))

        U = np.zeros((3, 3))
        U[:, 1:3] = U0[:, 1:3]
        ringtmp = U.dot(ringpts)

        # Compute touching circular paths
        phi = np.linspace(0, np.pi, n_points)
        pts = np.vstack((np.cos(phi), np.sin(phi), np.zeros((1, len(phi)))))

        xring = np.zeros((n_rings, n_points, 3))
        for j in range(n_rings):
                U = np.zeros((3, 3))
                U[:, 0] = U0[:, 0]
                U[:, 1] = ringtmp[:, j]
                xring[j, :] = (U.dot(pts).T + mean)

        # Reshape points in 2 dimensional array:
        points = xring.reshape((n_rings * n_points, 3))

        # Compute triangle points:
        triangles = tri_elipsoid(n_rings, n_points)

        # Plot surface:
        ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                                        triangles=triangles, linewidth=linewidth, alpha=alpha, color=color,
                                        edgecolor=color)


def plot_gmm3d(ax, means, covars, n_points=20, n_rings=15, color='red', alpha=0.4,
                           linewidth=0):
        ''' Plot 3D gmm '''
        n_states = means.shape[0]
        for i in range(n_states):
                print
                plot_gauss3d(ax, means[i,], covars[i,],
                                         n_points=n_points, n_rings=n_rings, color=color,
                                         alpha=alpha, linewidth=linewidth)

