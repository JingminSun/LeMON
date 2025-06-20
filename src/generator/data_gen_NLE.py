#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     advection_multi_solution_Hydra.py
  Authors:  Makoto Takamoto (makoto.takamoto@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
from math import ceil, log, exp

import os

# os.environ['JAX_PLATFORM_NAME'] = 'gpu'
import matplotlib.pyplot as plt
import seaborn as sns

import jax

# print("Default JAX backend:", jax.default_backend())
from tqdm import tqdm

from jax import vmap
import jax.numpy as jnp
from jax import device_put, lax, random
import numpy as np
import torch
from functools import partial
from .gp_jax import rbf_kernel_jax, rbf_sin_kernel_jax, rbf_circle_kernel_jax
from .gp_jax import generate_gaussian_process as generate_gaussian_process_jax
from .utils_datagenNLE import (
    init_multi,
    Courant,
    bc,
    limiting,
    Courant_diff,
    Courant_HD,
    bc_HD,
    limiting_HD,
    Courant_vis_HD,
)
from .utils_datagenNLE import (
    init_multi_HD,
    init_multi_HD_shock,
    init_multi_HD_KH,
    init_multi_HD_2DTurb,
    init_multi_HD_2DRand,
    init_multi_HD_3DTurb,
    init_multi_HD_3DRand,
)
import time


def _pass(carry):
    return carry


# Init arguments with Hydra

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15,16))
def diff_react_1D_f(
    xR,
    xL,
    nx,
    ini_time,
    fin_time,
    dt_save,
    it_tot,
    CFL,
    numbers,
    show_steps,
    init_key,
    rho,
    nu,
    a=0,
    react_term="logistic",
    num_steps=10,
    IC_train=True,
    GivenIC = None,
):
    # basic parameters
    dx = (xR - xL) / nx
    dx_inv = 1.0 / dx

    # cell edge coordinate
    xe = jnp.linspace(xL, xR, nx + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * dx

    rho = rho
    nu = nu
    # jax.debug.print('rho: {}, nu: {}', rho, nu)

    # t-coordinate
    # it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    # tc = jnp.arange(it_tot + 1) * dt_save

    @jax.jit
    def evolve(u):
        t = ini_time
        tsave = t
        steps = 0
        i_save = 0
        dt = 0.0
        uu = jnp.zeros([it_tot, u.shape[0]])
        uu = uu.at[0].set(u)

        cond_fun = lambda x: x[0] < fin_time

        def _body_fun(carry):
            def _show(_carry):
                u, tsave, i_save, uu = _carry
                uu = uu.at[i_save].set(u)
                tsave += dt_save
                i_save += 1
                return (u, tsave, i_save, uu)

            t, tsave, steps, i_save, dt, u, uu = carry

            carry = (u, tsave, i_save, uu)
            u, tsave, i_save, uu = lax.cond(t >= tsave, _show, _pass, carry)

            carry = (u, t, dt, steps, tsave)
            u, t, dt, steps, tsave = lax.fori_loop(0, show_steps, simulation_fn, carry)

            return (t, tsave, steps, i_save, dt, u, uu)

        carry = t, tsave, steps, i_save, dt, u, uu
        t, tsave, steps, i_save, dt, u, uu = lax.while_loop(cond_fun, _body_fun, carry)
        uu = uu.at[-1].set(u)

        return uu

    @jax.jit
    def simulation_fn(i, carry):
        u, t, dt, steps, tsave = carry
        dt = Courant_diff(dx, nu) * CFL
        dt = jnp.min(jnp.array([dt, fin_time - t, tsave - t]))

        def _update(carry):
            u, dt = carry
            # preditor step for calculating t+dt/2-th time step
            u_tmp = update(u, u, dt * 0.5)
            # update using flux at t+dt/2-th time step
            u = update(u, u_tmp, dt)
            return u, dt

        carry = u, dt
        u, dt = lax.cond(dt > 1.0e-8, _update, _pass, carry)

        t += dt
        steps += 1
        return u, t, dt, steps, tsave

    @jax.jit
    def update(u, u_tmp, dt):
        # stiff part
        if react_term == "logistic":
            u = logistic_Solution(u, dt)
        elif react_term == "linear":
            u = linear_Solution(u, dt)
        elif react_term == "bistable":
            u = bistable_Solution(u, dt)
        elif react_term == "squarelogistic":
            u = squarelogistic_Solution(u, dt)
        else:
            raise NotImplementedError
        # diffusion
        f = flux(u_tmp)
        u -= dt * dx_inv * (f[1 : nx + 1] - f[0:nx])
        return u

    @jax.jit
    def flux(u):
        _u = bc(u, dx, Ncell=nx)  # index 2 for _U is equivalent with index 0 for u
        # 2nd-order diffusion flux
        f = -nu * (_u[2 : nx + 3] - _u[1 : nx + 2]) * dx_inv
        return f

    @jax.jit
    def logistic_Solution(u, dt):  # Piecewise_Exact_Solution method
        # stiff equation
        u = 1.0 / (1.0 + jnp.exp(-rho * dt) * (1.0 - u) / u)
        return u

    @jax.jit
    def linear_Solution(u, dt):  # Piecewise_Exact_Solution method
        # stiff equation
        u = u * jnp.exp(rho * dt)
        return u

    @jax.jit
    def bistable_Solution(u, dt):  # Piecewise_Exact_Solution method
        def euler_update(u, dt):
            """Perform one step of the Euler method."""
            du_dt = rho * u * (1 - u) * (u - a)
            return u + du_dt * dt

        for _ in range(num_steps):
            u = euler_update(u, dt / num_steps)
        return u

    @jax.jit
    def squarelogistic_Solution(u, dt):  # Piecewise_Exact_Solution method
        def euler_update(u, dt):
            """Perform one step of the Euler method."""
            du_dt = rho * (u**2) * (1 - u) ** 2
            return u + du_dt * dt

        for _ in range(num_steps):
            u = euler_update(u, dt / num_steps)
        return u

    # if IC_train == True or IC_train == False:
    #     if IC_train:
    #         u = init_multi(xc, numbers=numbers, k_tot=4, init_key=init_key, if_norm=True)
    #     else:
    #         u = generate_gaussian_process_jax(
    #             xc, init_key=init_key, num=numbers, kernel=rbf_kernel_jax, k_sigma=1, k_l=0.2
    #         )
    # else:
    #     u = IC_train
    if GivenIC is not None:
        u = GivenIC
    else:
        if IC_train:
                u = init_multi(xc, numbers=numbers, k_tot=4, init_key=init_key, if_norm=True)
        else:
                u = generate_gaussian_process_jax(
                    xc, init_key=init_key, num=numbers, kernel=rbf_kernel_jax, k_sigma=1, k_l=0.2
                )
    # slope = (u[:,-1] - u[:,0])/(xR-xL)
    #
    # u = u - jnp.outer(slope,xc)
    #
    # u = (u - jnp.min(u, axis=1).reshape(-1, 1)) / (
    #             jnp.max(u, axis=1).reshape(-1, 1) - jnp.min(u, axis=1).reshape(-1, 1))
    u = device_put(u)  # putting variables in GPU (not necessary??)

    # vm_evolve = vmap(evolve, 0, 0)
    # uu = vm_evolve(u)
    vm_evolve = jax.pmap(jax.vmap(evolve, axis_name="j"), axis_name="i")
    local_devices = jax.local_device_count()
    uu = vm_evolve(u.reshape([local_devices, numbers // local_devices, -1]))

    return uu.reshape([1,numbers,it_tot,nx])


def diff_react_1D(cfg):
    # basic parameters
    xR = cfg["xR"]
    xL = cfg["xL"]
    nx = cfg["nx"]
    ini_time = cfg["ini_time"]
    fin_time = cfg["fin_time"]
    dt_save = cfg["dt_save"]
    CFL = cfg["CFL"]
    numbers = cfg["numbers"]
    show_steps = cfg["show_steps"]
    if_rand_param = cfg["if_rand_param"]
    rho = cfg["rho"]
    nu = cfg["nu"]
    init_key = cfg["init_key"]

    return diff_react_1D_f(xR, xL, nx, ini_time, fin_time, dt_save, CFL, numbers, show_steps, init_key, rho, nu)


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14,15,17))
def burgers_f(
    xR,
    xL,
    nx,
    ini_time,
    fin_time,
    dt_save,
    it_tot,
    CFL,
    numbers,
    show_steps,
    init_key,
    epsilon,
    k,
    viscous=True,
    fluxx="quadratic",
    IC_train=True,
    GivenIC=None,
    mode = "periodic"
):
    # basic parameters
    pi_inv = 1.0 / jnp.pi
    dx = (xR - xL) / nx
    dx_inv = 1.0 / dx

    # cell edge coordinate
    xe = jnp.linspace(xL, xR, nx + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * dx

    epsilon = epsilon
    # jax.debug.print('epsilon:{} ', epsilon)
    # t-coordinate
    # it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    # tc = jnp.arange(it_tot + 1) * dt_save

    @jax.jit
    def evolve(u):
        t = ini_time
        tsave = t
        steps = 0
        i_save = 0
        dt = 0.0
        uu = jnp.zeros([it_tot, u.shape[0]])
        uu = uu.at[0].set(u)

        cond_fun = lambda x: x[0] < fin_time

        def _body_fun(carry):
            def _show(_carry):
                u, tsave, i_save, uu = _carry
                uu = uu.at[i_save].set(u)
                tsave += dt_save
                i_save += 1
                return (u, tsave, i_save, uu)

            t, tsave, steps, i_save, dt, u, uu = carry

            carry = (u, tsave, i_save, uu)
            u, tsave, i_save, uu = lax.cond(t >= tsave, _show, _pass, carry)

            carry = (u, t, dt, steps, tsave)
            u, t, dt, steps, tsave = lax.fori_loop(0, show_steps, simulation_fn, carry)

            return (t, tsave, steps, i_save, dt, u, uu)

        carry = t, tsave, steps, i_save, dt, u, uu
        t, tsave, steps, i_save, dt, u, uu = lax.while_loop(cond_fun, _body_fun, carry)
        uu = uu.at[-1].set(u)

        return uu

    @jax.jit
    def simulation_fn(i, carry):
        u, t, dt, steps, tsave = carry
        dt_adv = Courant(u, dx) * CFL
        dt_dif = Courant_diff(dx, epsilon * pi_inv) * CFL
        dt = jnp.min(jnp.array([dt_adv, dt_dif, fin_time - t, tsave - t]))

        def _update(carry):
            u, dt = carry
            # preditor step for calculating t+dt/2-th time step
            u_tmp = update(u, u, dt * 0.5)
            # update using flux at t+dt/2-th time step
            u = update(u, u_tmp, dt)
            return u, dt

        carry = u, dt
        u, dt = lax.cond(dt > 1.0e-8, _update, _pass, carry)

        t += dt
        steps += 1
        return u, t, dt, steps, tsave

    @jax.jit
    def update(u, u_tmp, dt):
        f = flux(u_tmp)
        u -= dt * dx_inv * (f[1 : nx + 1] - f[0:nx])
        return u

    def flux(u):
        _u = bc(u, dx, Ncell=nx,mode=mode)  # index 2 for _U is equivalent with index 0 for u
        uL, uR = limiting(_u, nx, if_second_order=1.0)
        if fluxx == "quadratic":
            fL = k * 0.5 * uL**2
            fR = k * 0.5 * uR**2
        elif fluxx == "linear":
            fL = k * uL
            fR = k * uR
        elif fluxx == "cubic":
            fL = k * uL**3 / 3
            fR = k * uR**3 / 3
        elif fluxx == "sin":
            fL = k * jnp.sin(uL)
            fR = k * jnp.sin(uR)
        elif fluxx == "cos":
            fL = k * jnp.cos(uL)
            fR = k * jnp.cos(uR)
        else:
            assert isinstance(fluxx,tuple)
            fL = 0
            fR = 0
            for flux in fluxx:
                if flux == "quadratic":
                    fL += k * 0.5 * uL ** 2
                    fR += k * 0.5 * uR ** 2
                elif flux == "linear":
                    fL += k * uL
                    fR += k * uR
                elif flux == "cubic":
                    fL += k * uL ** 3 / 3
                    fR += k * uR ** 3 / 3
                elif flux == "sin":
                    fL += k * jnp.sin(uL)
                    fR += k * jnp.sin(uR)
                elif flux == "cos":
                    fL += k * jnp.cos(uL)
                    fR += k * jnp.cos(uR)
                else:
                    raise "Wrong flux"
        # upwind advection scheme
        f_upwd = 0.5 * (
            fR[1 : nx + 2]
            + fL[2 : nx + 3]
            - 0.5 * jnp.abs(uL[2 : nx + 3] + uR[1 : nx + 2]) * (uL[2 : nx + 3] - uR[1 : nx + 2])
        )
        # source term
        if viscous:
            f_upwd += -epsilon * pi_inv * (_u[2 : nx + 3] - _u[1 : nx + 2]) * dx_inv
        return f_upwd
    if GivenIC is not None:
        u = GivenIC
    else:
        if IC_train:
                u = init_multi(xc, numbers=numbers, k_tot=4, init_key=init_key, if_norm=True)
        else:
                u = generate_gaussian_process_jax(
                    xc, init_key=init_key, num=numbers, kernel=rbf_kernel_jax, k_sigma=1, k_l=0.2
                )
    # slope = (u[:,-1] - u[:,0])/(xR-xL)
    #
    # u = u - jnp.outer(slope,xc)
    # u = (u - jnp.min(u, axis=1).reshape(-1, 1)) / (
    #             jnp.max(u, axis=1).reshape(-1, 1) - jnp.min(u, axis=1).reshape(-1, 1))
    u = device_put(u)  # putting variables in GPU (not necessary??)

    # vm_evolve = vmap(evolve, 0, 0)
    # uu = vm_evolve(u)
    vm_evolve = jax.pmap(jax.vmap(evolve, axis_name="j"), axis_name="i")
    local_devices = jax.local_device_count()
    uu = vm_evolve(u.reshape([local_devices, numbers // local_devices, -1]))
    return uu.reshape([1,numbers,it_tot,nx])


#
# def burgers(cfg):
#     # basic parameters
#     xR = cfg['xR']
#     xL = cfg['xL']
#     nx = cfg['nx']
#     ini_time = cfg['ini_time']
#     fin_time = cfg['fin_time']
#     dt_save = cfg['dt_save']
#     CFL = cfg['CFL']
#     numbers = cfg['numbers']
#     show_steps = cfg['show_steps']
#     epsilon = cfg['epsilon']
#     init_key = cfg['init_key']
#
#     return burgers_f(xR, xL, nx, ini_time, fin_time, dt_save, CFL, numbers, show_steps, init_key,epsilon)


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 21, 22, 23))
def CFD_f(
    xR,
    xL,
    yR,
    yL,
    zR,
    zL,
    nx,
    ny,
    nz,
    ini_time,
    fin_time,
    dt_save,
    it_tot,
    CFL,
    numbers,
    show_steps,
    init_key,
    gamma,
    bc,
    zeta,
    eta,
    if_second_order,
    init_mode_Multi,
    M0=0,
    dkMx=1.0,
    p_floor=1e-4,
    k_tot=4,
):
    # physical constants
    # gamma = cfg.args.gamma  # 3D non-relativistic gas
    gammi1 = gamma - 1.0
    gamminv1 = 1.0 / gammi1
    gamgamm1inv = gamma * gamminv1
    gammi1 = gamma - 1.0
    gampl1 = gamma + 1.0
    gammi3 = gamma - 3.0
    gampl3 = gamma + 3.0

    BCs = ["trans", "periodic", "KHI"]  # reflect
    assert bc in BCs, "bc should be in 'trans, reflect, periodic'"

    dx = (xR - xL) / nx
    dx_inv = 1.0 / dx
    #
    dy = (yR - yL) / ny
    dy_inv = 1.0 / dy
    #
    dz = (zR - zL) / nz
    dz_inv = 1.0 / dz

    # cell edge coordinate
    xe = jnp.linspace(xL, xR, nx + 1)
    ye = jnp.linspace(yL, yR, ny + 1)
    ze = jnp.linspace(zL, zR, nz + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * dx
    yc = ye[:-1] + 0.5 * dy
    zc = ze[:-1] + 0.5 * dz

    # # t-coordinate
    # it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    # tc = jnp.arange(it_tot + 1) * dt_save

    # jax.debug.print('gamma: {}', gamma)
    # jax.debug.print('zeta: {}, eta: {}',zeta, eta)
    visc = zeta + eta / 3.0

    def evolve(Q):
        t = ini_time
        tsave = t
        steps = 0
        i_save = 0
        dt = 0.0

        # tm_ini = time.time()

        DDD = jnp.zeros([it_tot, nx, ny, nz])
        VVx = jnp.zeros([it_tot, nx, ny, nz])
        VVy = jnp.zeros([it_tot, nx, ny, nz])
        VVz = jnp.zeros([it_tot, nx, ny, nz])
        PPP = jnp.zeros([it_tot, nx, ny, nz])
        # initial time-step
        DDD = DDD.at[0].set(Q[0, 2:-2, 2:-2, 2:-2])
        VVx = VVx.at[0].set(Q[1, 2:-2, 2:-2, 2:-2])
        VVy = VVy.at[0].set(Q[2, 2:-2, 2:-2, 2:-2])
        VVz = VVz.at[0].set(Q[3, 2:-2, 2:-2, 2:-2])
        PPP = PPP.at[0].set(Q[4, 2:-2, 2:-2, 2:-2])

        cond_fun = lambda x: x[0] < fin_time

        def _body_fun(carry):
            def _save(_carry):
                Q, tsave, i_save, DDD, VVx, VVy, VVz, PPP = _carry

                DDD = DDD.at[i_save].set(Q[0, 2:-2, 2:-2, 2:-2])
                VVx = VVx.at[i_save].set(Q[1, 2:-2, 2:-2, 2:-2])
                VVy = VVy.at[i_save].set(Q[2, 2:-2, 2:-2, 2:-2])
                VVz = VVz.at[i_save].set(Q[3, 2:-2, 2:-2, 2:-2])
                PPP = PPP.at[i_save].set(Q[4, 2:-2, 2:-2, 2:-2])

                tsave += dt_save
                i_save += 1
                return (Q, tsave, i_save, DDD, VVx, VVy, VVz, PPP)

            t, tsave, steps, i_save, dt, Q, DDD, VVx, VVy, VVz, PPP = carry

            # if save data
            carry = (Q, tsave, i_save, DDD, VVx, VVy, VVz, PPP)
            Q, tsave, i_save, DDD, VVx, VVy, VVz, PPP = lax.cond(t >= tsave, _save, _pass, carry)

            carry = (Q, t, dt, steps, tsave)
            Q, t, dt, steps, tsave = lax.fori_loop(0, show_steps, simulation_fn, carry)

            return (t, tsave, steps, i_save, dt, Q, DDD, VVx, VVy, VVz, PPP)

        carry = t, tsave, steps, i_save, dt, Q, DDD, VVx, VVy, VVz, PPP
        t, tsave, steps, i_save, dt, Q, DDD, VVx, VVy, VVz, PPP = lax.while_loop(cond_fun, _body_fun, carry)

        # tm_fin = time.time()
        # print('total elapsed time is {} sec'.format(tm_fin - tm_ini))
        DDD = DDD.at[-1].set(Q[0, 2:-2, 2:-2, 2:-2])
        VVx = VVx.at[-1].set(Q[1, 2:-2, 2:-2, 2:-2])
        VVy = VVy.at[-1].set(Q[2, 2:-2, 2:-2, 2:-2])
        VVz = VVz.at[-1].set(Q[3, 2:-2, 2:-2, 2:-2])
        PPP = PPP.at[-1].set(Q[4, 2:-2, 2:-2, 2:-2])
        return t, DDD, VVx, VVy, VVz, PPP

    @jax.jit
    def simulation_fn(i, carry):
        Q, t, dt, steps, tsave = carry
        dt = Courant_HD(Q[:, 2:-2, 2:-2, 2:-2], dx, dy, dz, gamma) * CFL
        dt = jnp.min(jnp.array([dt, fin_time - t, tsave - t]))

        def _update(carry):
            Q, dt = carry

            # preditor step for calculating t+dt/2-th time step
            Q_tmp = bc_HD(Q, mode=bc)  # index 2 for _U is equivalent with index 0 for u
            Q_tmp = update(Q, Q_tmp, dt * 0.5)
            # update using flux at t+dt/2-th time step
            Q_tmp = bc_HD(Q_tmp, mode=bc)  # index 2 for _U is equivalent with index 0 for u
            Q = update(Q, Q_tmp, dt)

            # update via viscosity
            # d_min = jnp.min(Q[0])
            # dt_vis = Courant_vis_HD(dx, dy, dz, eta/d_min, zeta/d_min) * cfg.args.CFL  # for realistic viscosity

            dt_vis = Courant_vis_HD(dx, dy, dz, eta, zeta) * CFL
            dt_vis = jnp.min(jnp.array([dt_vis, dt]))
            t_vis = 0.0

            carry = Q, dt, dt_vis, t_vis
            Q, dt, dt_vis, t_vis = lax.while_loop(lambda x: x[1] - x[3] > 1.0e-8, update_vis, carry)
            return Q, dt

        carry = Q, dt
        Q, dt = lax.cond(dt > 1.0e-8, _update, _pass, carry)

        t += dt
        steps += 1
        return Q, t, dt, steps, tsave

    @jax.jit
    def update(Q, Q_tmp, dt):
        # calculate conservative variables
        D0 = Q[0]
        Mx = Q[1] * Q[0]
        My = Q[2] * Q[0]
        Mz = Q[3] * Q[0]
        E0 = Q[4] * gamminv1 + 0.5 * (Mx * Q[1] + My * Q[2] + Mz * Q[3])

        D0 = D0[2:-2, 2:-2, 2:-2]
        Mx = Mx[2:-2, 2:-2, 2:-2]
        My = My[2:-2, 2:-2, 2:-2]
        Mz = Mz[2:-2, 2:-2, 2:-2]
        E0 = E0[2:-2, 2:-2, 2:-2]

        # calculate flux
        fx = flux_x(Q_tmp)
        fy = flux_y(Q_tmp)
        fz = flux_z(Q_tmp)

        # update conservative variables
        dtdx, dtdy, dtdz = dt * dx_inv, dt * dy_inv, dt * dz_inv
        D0 -= (
            dtdx * (fx[0, 1:, 2:-2, 2:-2] - fx[0, :-1, 2:-2, 2:-2])
            + dtdy * (fy[0, 2:-2, 1:, 2:-2] - fy[0, 2:-2, :-1, 2:-2])
            + dtdz * (fz[0, 2:-2, 2:-2, 1:] - fz[0, 2:-2, 2:-2, :-1])
        )

        Mx -= (
            dtdx * (fx[1, 1:, 2:-2, 2:-2] - fx[1, :-1, 2:-2, 2:-2])
            + dtdy * (fy[1, 2:-2, 1:, 2:-2] - fy[1, 2:-2, :-1, 2:-2])
            + dtdz * (fz[1, 2:-2, 2:-2, 1:] - fz[1, 2:-2, 2:-2, :-1])
        )

        My -= (
            dtdx * (fx[2, 1:, 2:-2, 2:-2] - fx[2, :-1, 2:-2, 2:-2])
            + dtdy * (fy[2, 2:-2, 1:, 2:-2] - fy[2, 2:-2, :-1, 2:-2])
            + dtdz * (fz[2, 2:-2, 2:-2, 1:] - fz[2, 2:-2, 2:-2, :-1])
        )

        Mz -= (
            dtdx * (fx[3, 1:, 2:-2, 2:-2] - fx[3, :-1, 2:-2, 2:-2])
            + dtdy * (fy[3, 2:-2, 1:, 2:-2] - fy[3, 2:-2, :-1, 2:-2])
            + dtdz * (fz[3, 2:-2, 2:-2, 1:] - fz[3, 2:-2, 2:-2, :-1])
        )

        E0 -= (
            dtdx * (fx[4, 1:, 2:-2, 2:-2] - fx[4, :-1, 2:-2, 2:-2])
            + dtdy * (fy[4, 2:-2, 1:, 2:-2] - fy[4, 2:-2, :-1, 2:-2])
            + dtdz * (fz[4, 2:-2, 2:-2, 1:] - fz[4, 2:-2, 2:-2, :-1])
        )

        # reverse primitive variables
        Q = Q.at[0, 2:-2, 2:-2, 2:-2].set(D0)  # d
        Q = Q.at[1, 2:-2, 2:-2, 2:-2].set(Mx / D0)  # vx
        Q = Q.at[2, 2:-2, 2:-2, 2:-2].set(My / D0)  # vy
        Q = Q.at[3, 2:-2, 2:-2, 2:-2].set(Mz / D0)  # vz
        Q = Q.at[4, 2:-2, 2:-2, 2:-2].set(gammi1 * (E0 - 0.5 * (Mx**2 + My**2 + Mz**2) / D0))  # p
        Q = Q.at[4].set(jnp.where(Q[4] > 1.0e-8, Q[4], p_floor))

        return Q

    @jax.jit
    def update_vis(carry):
        def _update_vis_x(carry):
            Q, dt = carry
            # calculate conservative variables
            D0 = Q[0]
            Mx = Q[1] * D0
            My = Q[2] * D0
            Mz = Q[3] * D0
            E0 = Q[4] * gamminv1 + 0.5 * (Mx * Q[1] + My * Q[2] + Mz * Q[3])

            # calculate flux
            dtdx = dt * dx_inv
            # here the viscosity is eta*D0, so that dv/dt = eta*d^2v/dx^2 (not realistic viscosity but fast to calculate)
            Dm = 0.5 * (D0[2:-1, 2:-2, 2:-2] + D0[1:-2, 2:-2, 2:-2])

            fMx = (eta + visc) * Dm * dx_inv * (Q[1, 2:-1, 2:-2, 2:-2] - Q[1, 1:-2, 2:-2, 2:-2])
            fMy = eta * Dm * dx_inv * (Q[2, 2:-1, 2:-2, 2:-2] - Q[2, 1:-2, 2:-2, 2:-2])
            fMz = eta * Dm * dx_inv * (Q[3, 2:-1, 2:-2, 2:-2] - Q[3, 1:-2, 2:-2, 2:-2])
            fE = 0.5 * (eta + visc) * Dm * dx_inv * (
                Q[1, 2:-1, 2:-2, 2:-2] ** 2 - Q[1, 1:-2, 2:-2, 2:-2] ** 2
            ) + 0.5 * eta * Dm * dx_inv * (
                (Q[2, 2:-1, 2:-2, 2:-2] ** 2 - Q[2, 1:-2, 2:-2, 2:-2] ** 2)
                + (Q[3, 2:-1, 2:-2, 2:-2] ** 2 - Q[3, 1:-2, 2:-2, 2:-2] ** 2)
            )

            D0 = D0[2:-2, 2:-2, 2:-2]
            Mx = Mx[2:-2, 2:-2, 2:-2]
            My = My[2:-2, 2:-2, 2:-2]
            Mz = Mz[2:-2, 2:-2, 2:-2]
            E0 = E0[2:-2, 2:-2, 2:-2]

            Mx += dtdx * (fMx[1:, :, :] - fMx[:-1, :, :])
            My += dtdx * (fMy[1:, :, :] - fMy[:-1, :, :])
            Mz += dtdx * (fMz[1:, :, :] - fMz[:-1, :, :])
            E0 += dtdx * (fE[1:, :, :] - fE[:-1, :, :])

            # reverse primitive variables
            Q = Q.at[1, 2:-2, 2:-2, 2:-2].set(Mx / D0)  # vx
            Q = Q.at[2, 2:-2, 2:-2, 2:-2].set(My / D0)  # vy
            Q = Q.at[3, 2:-2, 2:-2, 2:-2].set(Mz / D0)  # vz
            Q = Q.at[4, 2:-2, 2:-2, 2:-2].set(gammi1 * (E0 - 0.5 * (Mx**2 + My**2 + Mz**2) / D0))  # p

            return Q, dt

        def _update_vis_y(carry):
            Q, dt = carry
            # calculate conservative variables
            D0 = Q[0]
            Mx = Q[1] * D0
            My = Q[2] * D0
            Mz = Q[3] * D0
            E0 = Q[4] * gamminv1 + 0.5 * (Mx * Q[1] + My * Q[2] + Mz * Q[3])

            # calculate flux
            dtdy = dt * dy_inv
            # here the viscosity is eta*D0, so that dv/dt = eta*d^2v/dx^2 (not realistic viscosity but fast to calculate)
            Dm = 0.5 * (D0[2:-2, 2:-1, 2:-2] + D0[2:-2, 1:-2, 2:-2])

            fMx = eta * Dm * dy_inv * (Q[1, 2:-2, 2:-1, 2:-2] - Q[1, 2:-2, 1:-2, 2:-2])
            fMy = (eta + visc) * Dm * dy_inv * (Q[2, 2:-2, 2:-1, 2:-2] - Q[2, 2:-2, 1:-2, 2:-2])
            fMz = eta * Dm * dy_inv * (Q[3, 2:-2, 2:-1, 2:-2] - Q[3, 2:-2, 1:-2, 2:-2])
            fE = 0.5 * (eta + visc) * Dm * dy_inv * (
                Q[2, 2:-2, 2:-1, 2:-2] ** 2 - Q[2, 2:-2, 1:-2, 2:-2] ** 2
            ) + 0.5 * eta * Dm * dy_inv * (
                (Q[3, 2:-2, 2:-1, 2:-2] ** 2 - Q[3, 2:-2, 1:-2, 2:-2] ** 2)
                + (Q[1, 2:-2, 2:-1, 2:-2] ** 2 - Q[1, 2:-2, 1:-2, 2:-2] ** 2)
            )

            D0 = D0[2:-2, 2:-2, 2:-2]
            Mx = Mx[2:-2, 2:-2, 2:-2]
            My = My[2:-2, 2:-2, 2:-2]
            Mz = Mz[2:-2, 2:-2, 2:-2]
            E0 = E0[2:-2, 2:-2, 2:-2]

            Mx += dtdy * (fMx[:, 1:, :] - fMx[:, :-1, :])
            My += dtdy * (fMy[:, 1:, :] - fMy[:, :-1, :])
            Mz += dtdy * (fMz[:, 1:, :] - fMz[:, :-1, :])
            E0 += dtdy * (fE[:, 1:, :] - fE[:, :-1, :])

            # reverse primitive variables
            Q = Q.at[1, 2:-2, 2:-2, 2:-2].set(Mx / D0)  # vx
            Q = Q.at[2, 2:-2, 2:-2, 2:-2].set(My / D0)  # vy
            Q = Q.at[3, 2:-2, 2:-2, 2:-2].set(Mz / D0)  # vz
            Q = Q.at[4, 2:-2, 2:-2, 2:-2].set(gammi1 * (E0 - 0.5 * (Mx**2 + My**2 + Mz**2) / D0))  # p

            return Q, dt

        def _update_vis_z(carry):
            Q, dt = carry
            # calculate conservative variables
            D0 = Q[0]
            Mx = Q[1] * D0
            My = Q[2] * D0
            Mz = Q[3] * D0
            E0 = Q[4] * gamminv1 + 0.5 * (Mx * Q[1] + My * Q[2] + Mz * Q[3])

            # calculate flux
            dtdz = dt * dz_inv
            # here the viscosity is eta*D0, so that dv/dt = eta*d^2v/dx^2 (not realistic viscosity but fast to calculate)
            Dm = 0.5 * (D0[2:-2, 2:-2, 2:-1] + D0[2:-2, 2:-2, 1:-2])

            fMx = eta * Dm * dz_inv * (Q[1, 2:-2, 2:-2, 2:-1] - Q[1, 2:-2, 2:-2, 1:-2])
            fMy = eta * Dm * dz_inv * (Q[2, 2:-2, 2:-2, 2:-1] - Q[2, 2:-2, 2:-2, 1:-2])
            fMz = (eta + visc) * Dm * dz_inv * (Q[3, 2:-2, 2:-2, 2:-1] - Q[3, 2:-2, 2:-2, 1:-2])
            fE = 0.5 * (eta + visc) * Dm * dz_inv * (
                Q[3, 2:-2, 2:-2, 2:-1] ** 2 - Q[3, 2:-2, 2:-2, 1:-2] ** 2
            ) + 0.5 * eta * Dm * dz_inv * (
                (Q[1, 2:-2, 2:-2, 2:-1] ** 2 - Q[1, 2:-2, 2:-2, 1:-2] ** 2)
                + (Q[2, 2:-2, 2:-2, 2:-1] ** 2 - Q[2, 2:-2, 2:-2, 1:-2] ** 2)
            )

            D0 = D0[2:-2, 2:-2, 2:-2]
            Mx = Mx[2:-2, 2:-2, 2:-2]
            My = My[2:-2, 2:-2, 2:-2]
            Mz = Mz[2:-2, 2:-2, 2:-2]
            E0 = E0[2:-2, 2:-2, 2:-2]

            Mx += dtdz * (fMx[:, :, 1:] - fMx[:, :, :-1])
            My += dtdz * (fMy[:, :, 1:] - fMy[:, :, :-1])
            Mz += dtdz * (fMz[:, :, 1:] - fMz[:, :, :-1])
            E0 += dtdz * (fE[:, :, 1:] - fE[:, :, :-1])

            # reverse primitive variables
            Q = Q.at[1, 2:-2, 2:-2, 2:-2].set(Mx / D0)  # vx
            Q = Q.at[2, 2:-2, 2:-2, 2:-2].set(My / D0)  # vy
            Q = Q.at[3, 2:-2, 2:-2, 2:-2].set(Mz / D0)  # vz
            Q = Q.at[4, 2:-2, 2:-2, 2:-2].set(gammi1 * (E0 - 0.5 * (Mx**2 + My**2 + Mz**2) / D0))  # p

            return Q, dt

        Q, dt, dt_vis, t_vis = carry
        Q = bc_HD(Q, mode=bc)  # index 2 for _U is equivalent with index 0 for u
        dt_ev = jnp.min(jnp.array([dt, dt_vis, dt - t_vis]))

        carry = Q, dt_ev
        # directional split
        carry = _update_vis_x(carry)  # x
        carry = _update_vis_y(carry)  # y
        Q, d_ev = _update_vis_z(carry)  # z

        t_vis += dt_ev

        return Q, dt, dt_vis, t_vis

    @jax.jit
    def flux_x(Q):
        QL, QR = limiting_HD(Q, if_second_order=if_second_order)
        # f_Riemann = HLL(QL, QR, direc=0)
        f_Riemann = HLLC(QL, QR, direc=0)
        return f_Riemann

    @jax.jit
    def flux_y(Q):
        _Q = jnp.transpose(Q, (0, 2, 3, 1))  # (y, z, x)
        QL, QR = limiting_HD(_Q, if_second_order=if_second_order)
        # f_Riemann = jnp.transpose(HLL(QL, QR, direc=1), (0, 3, 1, 2))  # (x,y,z) = (Z,X,Y)
        f_Riemann = jnp.transpose(HLLC(QL, QR, direc=1), (0, 3, 1, 2))  # (x,y,z) = (Z,X,Y)
        return f_Riemann

    @jax.jit
    def flux_z(Q):
        _Q = jnp.transpose(Q, (0, 3, 1, 2))  # (z, x, y)
        QL, QR = limiting_HD(_Q, if_second_order=if_second_order)
        # f_Riemann = jnp.transpose(HLL(QL, QR, direc=2), (0, 2, 3, 1))
        f_Riemann = jnp.transpose(HLLC(QL, QR, direc=2), (0, 2, 3, 1))
        return f_Riemann

    @partial(jax.jit, static_argnums=(2,))
    def HLL(QL, QR, direc):
        # direc = 0, 1, 2: (X, Y, Z)
        iX, iY, iZ = direc + 1, (direc + 1) % 3 + 1, (direc + 2) % 3 + 1
        cfL = jnp.sqrt(gamma * QL[4] / QL[0])
        cfR = jnp.sqrt(gamma * QR[4] / QR[0])
        Sfl = jnp.minimum(QL[iX, 2:-1], QR[iX, 1:-2]) - jnp.maximum(cfL[2:-1], cfR[1:-2])  # left-going wave
        Sfr = jnp.maximum(QL[iX, 2:-1], QR[iX, 1:-2]) + jnp.maximum(cfL[2:-1], cfR[1:-2])  # right-going wave
        dcfi = 1.0 / (Sfr - Sfl + 1.0e-8)

        UL, UR = jnp.zeros_like(QL), jnp.zeros_like(QR)
        UL = UL.at[0].set(QL[0])
        UL = UL.at[iX].set(QL[0] * QL[iX])
        UL = UL.at[iY].set(QL[0] * QL[iY])
        UL = UL.at[iZ].set(QL[0] * QL[iZ])
        UL = UL.at[4].set(gamminv1 * QL[4] + 0.5 * (UL[iX] * QL[iX] + UL[iY] * QL[iY] + UL[iZ] * QL[iZ]))
        UR = UR.at[0].set(QR[0])
        UR = UR.at[iX].set(QR[0] * QR[iX])
        UR = UR.at[iY].set(QR[0] * QR[iY])
        UR = UR.at[iZ].set(QR[0] * QR[iZ])
        UR = UR.at[4].set(gamminv1 * QR[4] + 0.5 * (UR[iX] * QR[iX] + UR[iY] * QR[iY] + UR[iZ] * QR[iZ]))

        fL, fR = jnp.zeros_like(QL), jnp.zeros_like(QR)
        fL = fL.at[0].set(UL[iX])
        fL = fL.at[iX].set(UL[iX] * QL[iX] + QL[4])
        fL = fL.at[iY].set(UL[iX] * QL[iY])
        fL = fL.at[iZ].set(UL[iX] * QL[iZ])
        fL = fL.at[4].set((UL[4] + QL[4]) * QL[iX])
        fR = fR.at[0].set(UR[iX])
        fR = fR.at[iX].set(UR[iX] * QR[iX] + QR[4])
        fR = fR.at[iY].set(UR[iX] * QR[iY])
        fR = fR.at[iZ].set(UR[iX] * QR[iZ])
        fR = fR.at[4].set((UR[4] + QR[4]) * QR[iX])
        # upwind advection scheme
        fHLL = dcfi * (Sfr * fR[:, 1:-2] - Sfl * fL[:, 2:-1] + Sfl * Sfr * (UL[:, 2:-1] - UR[:, 1:-2]))

        # L: left of cell = right-going,  R: right of cell: left-going
        f_Riemann = jnp.where(Sfl > 0.0, fR[:, 1:-2], fHLL)
        f_Riemann = jnp.where(Sfr < 0.0, fL[:, 2:-1], f_Riemann)

        return f_Riemann

    @partial(jax.jit, static_argnums=(2,))
    def HLLC(QL, QR, direc):
        """full-Godunov method -- exact shock solution"""

        iX, iY, iZ = direc + 1, (direc + 1) % 3 + 1, (direc + 2) % 3 + 1
        cfL = jnp.sqrt(gamma * QL[4] / QL[0])
        cfR = jnp.sqrt(gamma * QR[4] / QR[0])
        Sfl = jnp.minimum(QL[iX, 2:-1], QR[iX, 1:-2]) - jnp.maximum(cfL[2:-1], cfR[1:-2])  # left-going wave
        Sfr = jnp.maximum(QL[iX, 2:-1], QR[iX, 1:-2]) + jnp.maximum(cfL[2:-1], cfR[1:-2])  # right-going wave

        UL, UR = jnp.zeros_like(QL), jnp.zeros_like(QR)
        UL = UL.at[0].set(QL[0])
        UL = UL.at[iX].set(QL[0] * QL[iX])
        UL = UL.at[iY].set(QL[0] * QL[iY])
        UL = UL.at[iZ].set(QL[0] * QL[iZ])
        UL = UL.at[4].set(gamminv1 * QL[4] + 0.5 * (UL[iX] * QL[iX] + UL[iY] * QL[iY] + UL[iZ] * QL[iZ]))
        UR = UR.at[0].set(QR[0])
        UR = UR.at[iX].set(QR[0] * QR[iX])
        UR = UR.at[iY].set(QR[0] * QR[iY])
        UR = UR.at[iZ].set(QR[0] * QR[iZ])
        UR = UR.at[4].set(gamminv1 * QR[4] + 0.5 * (UR[iX] * QR[iX] + UR[iY] * QR[iY] + UR[iZ] * QR[iZ]))

        Va = (Sfr - QL[iX, 2:-1]) * UL[iX, 2:-1] - (Sfl - QR[iX, 1:-2]) * UR[iX, 1:-2] - QL[4, 2:-1] + QR[4, 1:-2]
        Va /= (Sfr - QL[iX, 2:-1]) * QL[0, 2:-1] - (Sfl - QR[iX, 1:-2]) * QR[0, 1:-2]
        Pa = QR[4, 1:-2] + QR[0, 1:-2] * (Sfl - QR[iX, 1:-2]) * (Va - QR[iX, 1:-2])

        # shock jump condition
        Dal = QR[0, 1:-2] * (Sfl - QR[iX, 1:-2]) / (Sfl - Va)  # right-hand density
        Dar = QL[0, 2:-1] * (Sfr - QL[iX, 2:-1]) / (Sfr - Va)  # left-hand density

        fL, fR = jnp.zeros_like(QL), jnp.zeros_like(QR)
        fL = fL.at[0].set(UL[iX])
        fL = fL.at[iX].set(UL[iX] * QL[iX] + QL[4])
        fL = fL.at[iY].set(UL[iX] * QL[iY])
        fL = fL.at[iZ].set(UL[iX] * QL[iZ])
        fL = fL.at[4].set((UL[4] + QL[4]) * QL[iX])
        fR = fR.at[0].set(UR[iX])
        fR = fR.at[iX].set(UR[iX] * QR[iX] + QR[4])
        fR = fR.at[iY].set(UR[iX] * QR[iY])
        fR = fR.at[iZ].set(UR[iX] * QR[iZ])
        fR = fR.at[4].set((UR[4] + QR[4]) * QR[iX])
        # upwind advection scheme
        far, fal = jnp.zeros_like(QL[:, 2:-1]), jnp.zeros_like(QR[:, 1:-2])
        far = far.at[0].set(Dar * Va)
        far = far.at[iX].set(Dar * Va**2 + Pa)
        far = far.at[iY].set(Dar * Va * QL[iY, 2:-1])
        far = far.at[iZ].set(Dar * Va * QL[iZ, 2:-1])
        far = far.at[4].set((gamgamm1inv * Pa + 0.5 * Dar * (Va**2 + QL[iY, 2:-1] ** 2 + QL[iZ, 2:-1] ** 2)) * Va)
        fal = fal.at[0].set(Dal * Va)
        fal = fal.at[iX].set(Dal * Va**2 + Pa)
        fal = fal.at[iY].set(Dal * Va * QR[iY, 1:-2])
        fal = fal.at[iZ].set(Dal * Va * QR[iZ, 1:-2])
        fal = fal.at[4].set((gamgamm1inv * Pa + 0.5 * Dal * (Va**2 + QR[iY, 1:-2] ** 2 + QR[iZ, 1:-2] ** 2)) * Va)

        f_Riemann = jnp.where(Sfl > 0.0, fR[:, 1:-2], fL[:, 2:-1])  # Sf2 > 0 : supersonic
        f_Riemann = jnp.where(Sfl * Va < 0.0, fal, f_Riemann)  # SL < 0 and Va > 0 : sub-sonic
        f_Riemann = jnp.where(Sfr * Va < 0.0, far, f_Riemann)  # Va < 0 and SR > 0 : sub-sonic
        # f_Riemann = jnp.where(Sfr < 0., fL[:, 2:-1], f_Riemann) # SR < 0 : supersonic

        return f_Riemann

    Q = jnp.zeros([numbers, 5, nx + 4, ny + 4, nz + 4])
    if init_mode_Multi == "1D_rand":
        Q = Q.at[:, 0, 2:-2, 2:-2, 2:-2].set(
            init_multi_HD(
                xc,
                yc,
                zc,
                numbers=numbers,
                k_tot=3,
                init_key=init_key,
                num_choise_k=2,
                umin=1.0e0,
                umax=1.0e1,
                if_renorm=True,
            )
        )
        Q = Q.at[:, 4, 2:-2, 2:-2, 2:-2].set(
            init_multi_HD(
                xc,
                yc,
                zc,
                numbers=numbers,
                k_tot=3,
                init_key=init_key + 1,
                num_choise_k=2,
                umin=1.0e1,
                umax=1.0e2,
                if_renorm=True,
            )
        )
        Q = Q.at[:, 1, 2:-2, 2:-2, 2:-2].set(
            init_multi_HD(xc, yc, zc, numbers=numbers, k_tot=3, init_key=init_key + 2, num_choise_k=2, if_renorm=False)
        )
    elif init_mode_Multi == "1D_shocks":
        Q = Q.at[:, 0, 2:-2, 2:-2, 2:-2].set(
            init_multi_HD_shock(xc, yc, zc, numbers=numbers, init_key=init_key, umin=1.0e0, umax=1.0e1)
        )
        Q = Q.at[:, 4, 2:-2, 2:-2, 2:-2].set(
            init_multi_HD_shock(xc, yc, zc, numbers=numbers, init_key=init_key + 1, umin=1.0e1, umax=1.0e2)
        )
        Q = Q.at[:, 1, 2:-2, 2:-2, 2:-2].set(
            init_multi_HD_shock(xc, yc, zc, numbers=numbers, init_key=init_key + 2, umin=-0.5e0, umax=0.5e0)
        )
    elif init_mode_Multi == "KHs":
        assert 2.0 * yc[0] - (yc[1] - yc[0]) == 0.0, "yL is assumed 0!"
        print("now we are coming into KHs...")
        Q = init_multi_HD_KH(Q, xc, yc, zc, numbers=numbers, init_key=init_key, M0=M0, dkMx=dkMx, gamma=gamma)
    elif init_mode_Multi == "2D_Turbs":
        print("now we are coming into 2DTurbs......")
        Q = init_multi_HD_2DTurb(Q, xc, yc, zc, numbers=numbers, init_key=init_key, M0=M0, k_tot=k_tot, gamma=gamma)
    elif init_mode_Multi == "2D_rand":
        assert xe[0] == 0.0 and ye[0] == 0.0 and xe[-1] == 1.0 and ye[-1] == 1.0, "xc, yc should be between 0 and 1!"
        print("now we are coming into 2Drand......")
        Q = init_multi_HD_2DRand(Q, xc, yc, zc, numbers=numbers, init_key=init_key, M0=M0, k_tot=k_tot, gamma=gamma)
    elif init_mode_Multi == "3D_Turbs":
        print("now we are coming into 3DTurbs......")
        Q = init_multi_HD_3DTurb(Q, xc, yc, zc, numbers=numbers, init_key=init_key, M0=M0, k_tot=k_tot, gamma=gamma)
    elif init_mode_Multi == "3D_rand":
        print("now we are coming into 3Drand......")
        Q = init_multi_HD_3DRand(Q, xc, yc, zc, numbers=numbers, init_key=init_key, M0=M0, k_tot=k_tot, gamma=gamma)
    # jax.debug.print('initial conditions were prepared!!')
    Q = device_put(Q)  # putting variables in GPU (not necessary??)

    local_device_count = jax.local_device_count()
    pm_evolve = jax.pmap(jax.vmap(evolve, axis_name="j"), axis_name="i")
    t, DDD, VVx, VVy, VVz, PPP = pm_evolve(
        Q.reshape([local_device_count, numbers // local_device_count, 5, nx + 4, ny + 4, nz + 4])
    )

    itot = DDD.shape[2]
    DDD = DDD.reshape(numbers, itot, nx, ny, nz)
    VVx = VVx.reshape(numbers, itot, nx, ny, nz)
    VVy = VVy.reshape(numbers, itot, nx, ny, nz)
    VVz = VVz.reshape(numbers, itot, nx, ny, nz)
    PPP = PPP.reshape(numbers, itot, nx, ny, nz)

    return DDD, VVx, VVy, VVz, PPP
    # print('now data saving...')
    # jnp.save(cfg.args.save+'HD_Sols_'+cfg.args.init_mode_Multi+'_Eta'+str(eta)[:5]+'_Zeta'+str(zeta)[:5]+'_M'+str(cfg.args.M0)+'_key'+str(cfg.args.init_key)+'_D', DDD)
    # jnp.save(cfg.args.save+'HD_Sols_'+cfg.args.init_mode_Multi+'_Eta'+str(eta)[:5]+'_Zeta'+str(zeta)[:5]+'_M'+str(cfg.args.M0)+'_key'+str(cfg.args.init_key)+'_Vx', VVx)
    # jnp.save(cfg.args.save+'HD_Sols_'+cfg.args.init_mode_Multi+'_Eta'+str(eta)[:5]+'_Zeta'+str(zeta)[:5]+'_M'+str(cfg.args.M0)+'_key'+str(cfg.args.init_key)+'_Vy', VVy)
    # jnp.save(cfg.args.save+'HD_Sols_'+cfg.args.init_mode_Multi+'_Eta'+str(eta)[:5]+'_Zeta'+str(zeta)[:5]+'_M'+str(cfg.args.M0)+'_key'+str(cfg.args.init_key)+'_Vz', VVz)
    # jnp.save(cfg.args.save+'HD_Sols_'+cfg.args.init_mode_Multi+'_Eta'+str(eta)[:5]+'_Zeta'+str(zeta)[:5]+'_M'+str(cfg.args.M0)+'_key'+str(cfg.args.init_key)+'_P', PPP)
    # jnp.save(cfg.args.save + '/x_coordinate', xc)
    # jnp.save(cfg.args.save + '/y_coordinate', yc)
    # jnp.save(cfg.args.save + '/z_coordinate', zc)
    # jnp.save(cfg.args.save + '/t_coordinate', tc)


def CFD_1D(cfg):
    xR = cfg["xR"]
    xL = cfg["xL"]
    nx = cfg["nx"]
    yR = cfg["yR"]
    yL = cfg["yL"]
    ny = cfg["ny"]
    zR = cfg["zR"]
    zL = cfg["zL"]
    nz = cfg["nz"]
    ini_time = cfg["ini_time"]
    fin_time = cfg["fin_time"]
    dt_save = cfg["dt_save"]
    CFL = cfg["CFL"]
    numbers = cfg["numbers"]
    show_steps = cfg["show_steps"]
    init_key = cfg["init_key"]
    gamma = cfg["gamma"]
    bc = cfg["bc"]
    zeta = cfg["zeta"]
    eta = cfg["eta"]
    if_second_order = cfg["if_second_order"]
    init_mode_Multi = cfg["init_mode_Multi"]

    return CFD_f(
        xR,
        xL,
        yR,
        yL,
        zR,
        zL,
        nx,
        ny,
        nz,
        ini_time,
        fin_time,
        dt_save,
        CFL,
        numbers,
        show_steps,
        init_key,
        gamma,
        bc,
        zeta,
        eta,
        if_second_order,
        init_mode_Multi,
    )


# if __name__ == "__main__":
#     configadv = {
#         "beta": 1,
#         "xL": 0.0,
#         "xR": 2.0,
#         "nx": 64,
#         "ini_time": 0.0,
#         "fin_time": 6.0,
#         "dt_save": 6.0 / (192 - 1),
#         "CFL": 0.4,
#         "if_show": 1,
#         "numbers": 1,
#         "if_second_order": 1.0,
#         "show_steps": 100,
#         "init_key": 492802927,
#         "if_rand_param": False,
#     }
#
#     configdiff = {
#         "rho": 0.01,
#         "nu": 0.01,
#         "xL": 0.0,
#         "xR": 2.0,
#         "nx": 64,
#         "ini_time": 0.0,
#         "fin_time": 6.0,
#         "dt_save": 6.0 / (192 - 1),
#         "CFL": 0.4,
#         "numbers": 4,
#         "show_steps": 192,
#         "init_key": 492802927,
#         "if_rand_param": False,
#     }
#
#     configburgers = {
#         "epsilon": 0.01,
#         "xL": 0.0,
#         "xR": 2.0,
#         "nx": 64,
#         "ini_time": 0.0,
#         "fin_time": 6.0,
#         "dt_save": 6.0 / (192 - 1),
#         "CFL": 0.4,
#         "numbers": 4,
#         "show_steps": 192,
#         "init_key": 492802927,
#         "if_rand_param": False,
#     }
#
#     configCFD1D = {
#         "dt_save": 6.0 / (192 - 1),
#         "ini_time": 0.0,
#         "fin_time": 6.0,
#         "nx": 64,
#         "ny": 1,
#         "nz": 1,
#         "xL": -1.0,
#         "xR": 1.0,
#         "yL": -1.0,
#         "yR": 1.0,
#         "zL": -1.0,
#         "zR": 1.0,
#         "bc": "periodic",
#         "gamma": 1.6666666666666667,
#         "eta": 1.0e-2,
#         "zeta": 1.0e-2,
#         "CFL": 3.0e-1,
#         "if_second_order": 1.0,
#         "if_show": 1,
#         "show_steps": 100,
#         "p_floor": 1.0e-4,
#         "numbers": 4,
#         "init_mode_Multi": "1D_rand",
#         "init_key": 2022,
#         "if_rand_param": False,
#         "M0": 0.0,
#     }
#
#     configdiffreact = {
#         "rho": 0.1,
#         "nu": 5,
#         "xL": 0.0,
#         "xR": 2.0,
#         "nx": 64,
#         "ini_time": 0.0,
#         "fin_time": 6.0,
#         "dt_save": 6.0 / (192 - 1),
#         "CFL": 0.4,
#         "numbers": 4,
#         "show_steps": 192,
#         "init_key": 492802927,
#         "if_rand_param": False,
#     }
#     #
#     # uu = burgers(configburgers)
#     uu = adv_gen(configadv)
#     uuu = [torch.tensor(np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0)))) for i in range(uu.shape[1])]
#     uu_np = [np.array(jnp.transpose(uu[:, i, :, :], axes=(1, 2, 0))) for i in range(uu.shape[1])]
#
#     for i in range(uu.shape[1]):
#         sns.heatmap(uu_np[i][:, :, 0])
#         plt.xlabel("x")
#         plt.ylabel("t")
#         plt.show()
#
#     # D,V,_,_,P = CFD_1D(configCFD1D)
#     # uu =np.concatenate((D[:,:,:,:,0],V[:,:,:,:,0],P[:,:,:,0]), axis = 3)
#     # uuu = [torch.tensor(np.array(uu[i, :, :, :])) for i in range(uu.shape[0])]
#
#     # for i in tqdm(range(1000)):
#     #     uu = adv_gen(configadv)
#     #     uu =  [torch.tensor(np.array( jnp.transpose(uu[:,i, :, :], axes = (1,2,0)))) for i in range(uu.shape[1])]
#     #     # print(uu)
#     #
#     #     vv = diff_react_1D(configdiff)
#     #     vv =  [torch.tensor(np.array( jnp.transpose(vv[:,i, :, :], axes = (1,2,0)))) for i in range(vv.shape[1])]
#     #
#     #     ww = burgers(configburgers)
#     #     ww = [torch.tensor(np.array( jnp.transpose(ww[:,i, :, :], axes = (1,2,0)))) for i in range(ww.shape[1])]
#     #
#     #     D, V, _, _, P = CFD_1D(configCFD1D)
#     #     uu =np.concatenate((D[:,:,:,:,0],V[:,:,:,:,0],P[:,:,:,0]), axis = 3)
#     #     uuu = [torch.tensor(np.array(uu[i, :, :, :])) for i in range(uu.shape[0])]
#
