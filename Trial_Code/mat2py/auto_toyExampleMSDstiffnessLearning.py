import mat2py as mp
from mat2py.core import *


def main():
    close("all")
    clear("all")
    clc
    dataPath = "data/"
    load(M[[dataPath, "data_07.mat"]])
    path(path, "manifolds")
    path(path, "gmm_gmr")
    windowSize = 3
    convexOptim = 0
    spdApproxim = 1
    modelCHOL.nbStates = 4
    modelCHOL.dt = 0.02
    modelCHOL.nbSamples = 5
    reprosCHOL = M[6:8]
    time = M[(modelCHOL.dt) : (modelCHOL.dt) : 2]
    modelCHOL.params_diagRegFact = 1e-4
    modelPD.nbStates = 4
    modelPD.nbVar = 5
    modelPD.dt = 2e-2
    modelPD.Kp = 120
    modelPD.nbIter = 10
    modelPD.nbSamples = 5
    modelPD.nbIterEM = 10
    reprosMan = copy(reprosCHOL)
    modelPD.nbVarCovOut = (modelPD.nbVar) + (
        ((M[modelPD.nbVar]) @ ((modelPD.nbVar) - 1)) / 2
    )
    modelPD.params_diagRegFact = 1e-4
    clrmapCHOL_states = lines(modelCHOL.nbStates)
    clrmapMAN_states = lines(modelPD.nbStates)
    clrmap_demos = lines(modelCHOL.nbSamples)
    clrmap_summer = summer(max(M[[length(reprosCHOL), length(reprosMan)]]))
    demons = M[1:10]
    n_s_total = length(dD)
    n_s = modelCHOL.nbSamples
    posId = M[1:2]
    velId = M[3:4]
    accId = M[5:6]
    rdp = 2
    DAMPING = 1
    if DAMPING:
        D = 50
    else:
        D = 0
    LOAD_TOY_EX_DATA = 0
    if LOAD_TOY_EX_DATA:
        load(M[[dataPath, "toyExampleMSDstiffnessLearning.mat"]])
    for j in M[1:n_s_total]:
        sumF = 0
        for n in demons:
            dD(j).nbData = copy(nbData)
            sumF = sumF + dD(j).demo(n).DataF[I[:, :]]

        dD(j).avgFe = mrdivide(sumF, n)

    if _not(LOAD_TOY_EX_DATA):
        ww = diag(ones(2, 1) * 0.000)
        xIn[I[1, :]] = (M[1 : (nbData - rdp)]) @ (modelPD.dt)
        X = zeros(modelPD.nbVar, modelPD.nbVar, (M[nbData - rdp]) @ (modelPD.nbSamples))
        X[I[1, 1, :]] = reshape(
            repmat(xIn, 1, modelPD.nbSamples),
            1,
            1,
            (M[nbData - rdp]) @ (modelPD.nbSamples),
        )
        eeigU = zeros(2, nbData, n_s_total)
        eigVal = zeros(2, nbData, n_s_total)
        for j in M[1:n_s_total]:
            fprintf("Stiffness matrices estimation...\n")
            g = dD(j).xR1
            tt1 = 1
            tt2 = copy(windowSize)
            for n in demons:
                auxX = M[[]]
                for i in M[1:nbData]:
                    if ((i + windowSize) - 1) <= nbData:
                        auxX[I[:, :]] = repmat(g, 1, windowSize) - dD(j).demo(n).DataP(
                            posId, M[i : ((i + windowSize) - 1)]
                        )
                        auxY[I[:, :]] = (
                            dD(j).demo(n).DataP(accId, M[i : ((i + windowSize) - 1)])
                            + (
                                M[D]
                                @ dD(j)
                                .demo(n)
                                .DataP(velId, M[i : ((i + windowSize) - 1)])
                            )
                        ) - dD(j).demo(n).DataF[I[:, i : ((i + windowSize) - 1)]]
                    else:
                        auxX[I[:, :]] = repmat(g, 1, windowSize) - (
                            M[
                                [
                                    dD(j).demo(n).DataP[I[posId, i:end]],
                                    repmat(
                                        dD(j).demo(n).DataP[I[posId, end]],
                                        1,
                                        abs(((nbData - i) - windowSize) + 1),
                                    ),
                                ]
                            ]
                        )
                        auxY[I[:, :]] = (
                            (
                                M[
                                    [
                                        dD(j).demo(n).DataP[I[accId, i:end]],
                                        repmat(
                                            dD(j).demo(n).DataP[I[accId, end]],
                                            1,
                                            abs(((nbData - i) - windowSize) + 1),
                                        ),
                                    ]
                                ]
                            )
                            + (
                                (
                                    M[
                                        [
                                            dD(j).demo(n).DataP[I[velId, i:end]],
                                            repmat(
                                                dD(j).demo(n).DataP[I[velId, end]],
                                                1,
                                                abs(((nbData - i) - windowSize) + 1),
                                            ),
                                        ]
                                    ]
                                )
                                @ D
                            )
                        ) - (
                            M[
                                [
                                    dD(j).demo(n).DataF[I[:, i:end]],
                                    repmat(
                                        dD(j).demo(n).DataF[I[:, end]],
                                        1,
                                        abs(((nbData - i) - windowSize) + 1),
                                    ),
                                ]
                            ]
                        )
                    dD(j).xX[I[tt1:tt2, :, i]] = auxX[I[:, :]].H
                    dD(j).yY[I[tt1:tt2, :, i]] = auxY[I[:, :]].H

                tt1 = size(dD(j).xX[I[:, :, 1]], 1) + 1
                tt2 = size(dD(j).xX[I[:, :, 1]], 1) + windowSize

            for i in M[1:nbData]:
                XtX = (dD(j).xX[I[:, :, i]].H) @ dD(j).xX[I[:, :, i]]
                XtY = (dD(j).xX[I[:, :, i]].H) @ dD(j).yY[I[:, :, i]]
                if spdApproxim:
                    Korig = mldivide(XtX + ww, XtY)
                    eeigU[I[:, i, j]] = eig(Korig)
                    K = nearestSPD(Korig)
                if convexOptim:
                    Korig = mldivide(XtX + ww, XtY)
                    d = length(posId)
                    cvx_begin("sdp")
                    variable("Kcvx(d,d)", "symmetric")
                    minimize(norm((M[XtX] @ Kcvx) - XtY, 2))
                    Kcvx >= eye(d)
                    cvx_end
                    KpCVX[I[:, :, i, j]] = copy(Kcvx)
                    K = copy(Kcvx)
                vec, val = eig(K)
                if val(1, 1) < 1e-4:
                    val[I[1, 1]] = 1e-4
                    K = (M[vec] @ val) @ (vec.H)
                if val(2, 2) < 1e-4:
                    val[I[2, 2]] = 1e-4
                    K = (M[vec] @ val) @ (vec.H)
                Kp, p = chol(K)
                if p > 0:
                    Kp, pp = cholcov(K)
                eigVec[I[j]][I[:, :, i]] = copy(vec)
                eigVal[I[:, i, j]] = M[
                    val(1, 1),
                    val(2, 2),
                ]
                Kpt = zeros(3, 1)
                Kpt[I[1]] = Kp(1, 1)
                Kpt[I[2]] = Kp(1, 2)
                mm, nn = size(Kp)
                if (mm == 2) and (nn == 2):
                    Kpt[I[3]] = Kp(2, 2)
                if (mm == 1) and (nn == 2):
                    Kp[I[2, 1]] = 0
                    Kp[I[2, 2]] = 0
                dD(j).Kpt[I[:, i]] = copy(Kpt)
                dD(j).KP[I[:, :, i]] = copy(K)
                dD(j).KPXX[I[i]] = K(1, 1)
                dD(j).KPYY[I[i]] = K(2, 2)
                dD(j).KPXY[I[i]] = K(1, 2)
                dD(j).KPorig11[I[i]] = Korig(1, 1)
                dD(j).KPorig22[I[i]] = Korig(2, 2)
                dD(j).Kp[I[:, :, i]] = copy(Kp)
                dD(j).KO[I[:, :, i]] = M[
                    [],
                    [dD(j).KXX(i), dD(j).KXY(i)],
                    [dD(j).KYX(i), dD(j).KYY(i)],
                ]
                if (j <= (modelCHOL.nbSamples)) and (i > rdp):
                    X[I[4:5, 4:5, (i - rdp) + ((M[j - 1]) @ (nbData - rdp))]] = copy(K)
                    X[I[2:3, 2:3, (i - rdp) + ((M[j - 1]) @ (nbData - rdp))]] = diag(
                        dD(j).avgFe[I[:, i]]
                    )
                    dD(j).Data[I[2:3, (i - rdp) + ((nbData - rdp) @ (M[j - 1]))]] = dD(
                        j
                    ).avgFe[I[:, i]]
                    dD(j).Data[
                        I[4:6, (i - rdp) + ((nbData - rdp) @ (M[j - 1]))]
                    ] = copy(Kpt)
                    DataK[I[2:3, (i - rdp) + ((nbData - rdp) @ (M[j - 1]))]] = dD(
                        j
                    ).avgFe[I[:, i]]
                    DataK[I[4:6, (i - rdp) + ((nbData - rdp) @ (M[j - 1]))]] = copy(Kpt)

            if j <= 5:
                DataK[
                    I[
                        1,
                        (((nbData - rdp) @ (M[j - 1])) + 1) : (
                            (i - rdp) + ((nbData - rdp) @ (M[j - 1]))
                        ),
                    ]
                ] = time(M[1 : (nbData - rdp)])
                dD(j).Data[
                    I[
                        1,
                        (((nbData - rdp) @ (M[j - 1])) + 1) : (
                            (i - rdp) + ((nbData - rdp) @ (M[j - 1]))
                        ),
                    ]
                ] = time(M[1 : (nbData - rdp)])

    reprpduction = figure("position", M[[1000, 500, 500, 200]])
    hold("on")
    title("\fontsize{10}Cartesian trajectory of different MSD")
    for j in M[1:n_s]:
        m1 = 1
        dt = 0.02
        xr = M[[]]
        xR1 = copy(g)
        x1[I[:, 1]] = copy(xR1)
        dx1[I[:, 1]] = zeros(2, 1)
        ddx1[I[:, 1]] = zeros(2, 1)
        f1 = zeros(2, 1)
        Fe = dD(j).avgFe
        for i in M[2:nbData]:
            k1 = dD(j).KP[I[:, :, i]]
            f1[I[:, i - 1]] = M[k1] @ (x1[I[:, i - 1]] - xR1)
            ddx1[I[:, i]] = mrdivide(
                (Fe[I[:, i - 1]] - f1[I[:, i - 1]]) - (M[D] @ dx1[I[:, i - 1]]), m1
            )
            dx1[I[:, i]] = dx1[I[:, i - 1]] + (ddx1[I[:, i]] @ M[dt])
            x1[I[:, i]] = x1[I[:, i - 1]] + (dx1[I[:, i]] @ M[dt])

        n = 1
        for i in round(linspace(2, nbData - 2, 30)):
            x1x[I[:, n]] = x1[I[:, i]]
            n = n + 1

        plot(x1[I[1, :]], x1[I[2, :]])

    xlabel("$x$", "interpreter", "latex")
    ylabel("$y$", "interpreter", "latex")
    set(gca, "FontSize", 10, "TickLabelInterpreter", "latex")
    if _not(LOAD_TOY_EX_DATA):
        tic
        modelCHOL = init_GMM_kbins(DataK[I[:, :]], modelCHOL, modelCHOL.nbSamples)
        modelCHOL = EM_GMM(DataK[I[:, :]], modelCHOL)
        elapsed_t_CHOL[I[1]] = copy(toc)
        model_r = copy(modelCHOL)
    handel_GMM_GMR_CHOL = figure("position", M[[1000, 500, 600, 450]])
    hold("on")
    htt = subplot(4, 2, M[1:2])
    hold("on")
    t1 = title("\fontsize{10}GMM/GMR based on Cholesky decomposition")
    set(htt(1), "position", M[[0.1300, 0.95, 0.7750, 0.0]])
    htt.Visible = "off"
    t1.Visible = "on"
    subplot(3, 2, 1)
    hold("on")
    title("\fontsize{10}Demonstrated Stiffness")
    for n in M[1 : (modelCHOL.nbSamples)]:
        for i in round(linspace(1, nbData - rdp, 10)):
            plotGMM2(
                M[
                    i,
                    0,
                ],
                X(M[4:5], M[4:5], i + ((M[n - 1]) @ (nbData - rdp))),
                clrmap_demos[I[n, :]],
                0.4,
            )

    xlim(M[[-27, (nbData - rdp) + 10]])
    ylim(M[[-27, 27]])
    ylabel("$\mathbf{K}^p$", "Interpreter", "Latex", "Fontsize", 10)
    xlabel("$t$", "Fontsize", 10, "Interpreter", "Latex")
    set(gca, "FontSize", 10, "TickLabelInterpreter", "latex")
    subplot(3, 2, 2)
    hold("on")
    title("\fontsize{10}GMM centers on CHOL")
    for i in M[1 : (modelCHOL.nbStates)]:
        Mu[I[:, :, i]] = zeros(2, 2)
        Mu[I[1, 1, i]] = modelCHOL.Mu(4, i)
        Mu[I[1, 2, i]] = modelCHOL.Mu(5, i)
        Mu[I[2, 2, i]] = modelCHOL.Mu(6, i)
        Mu[I[:, :, i]] = (Mu[I[:, :, i]].H) @ Mu[I[:, :, i]]
        h, _ = plotGMM2(
            M[
                0,
                0,
            ],
            Mu[I[:, :, i]],
            clrmapMAN_states[I[i, :]],
            0.3,
        )
        hh[I[i]] = h(1)

    for i in round(linspace(nbData, 2 * nbData, 20)):
        Mu[I[:, :, i]] = zeros(2, 2)
        Mu[I[1, 1, i]] = DataK(4, i)
        Mu[I[1, 2, i]] = DataK(5, i)
        Mu[I[2, 2, i]] = DataK(6, i)
        Mu[I[:, :, i]] = (Mu[I[:, :, i]].H) @ Mu[I[:, :, i]]
        plotGMM2(
            M[
                50,
                0,
            ],
            Mu[I[:, :, i]],
            M[[0.6, 0.6, 0.6]],
            0.1,
        )

    ylabel("$\mathbf{K}^p$", "Interpreter", "Latex", "Fontsize", 10)
    axis("equal")
    xlim(M[[-25, 75]])
    ylim(M[[-25, 25]])
    set(gca, "FontSize", 10, "TickLabelInterpreter", "latex")
    subplot(3, 2, M[[3, 4]])
    hold("on")
    title("\fontsize{10}Demonstrated Stiffness and GMM centers")
    sc = mrdivide(1, modelCHOL.dt)
    for i in M[1 : size(DataK, 2)]:
        Mu[I[:, :, i]] = zeros(2, 2)
        Mu[I[1, 1, i]] = DataK(4, i)
        Mu[I[1, 2, i]] = DataK(5, i)
        Mu[I[2, 2, i]] = DataK(6, i)
        Mu[I[:, :, i]] = (Mu[I[:, :, i]].H) @ Mu[I[:, :, i]]
        plotGMM2(
            M[
                DataK(1, i) @ M[sc],
                0,
            ],
            Mu[I[:, :, i]],
            M[[0.6, 0.6, 0.6]],
            0.1,
        )

    for i in M[1 : (modelCHOL.nbStates)]:
        Mu[I[:, :, i]] = zeros(2, 2)
        Mu[I[1, 1, i]] = modelCHOL.Mu(4, i)
        Mu[I[1, 2, i]] = modelCHOL.Mu(5, i)
        Mu[I[2, 2, i]] = modelCHOL.Mu(6, i)
        Mu[I[:, :, i]] = (Mu[I[:, :, i]].H) @ Mu[I[:, :, i]]
        plotGMM2(
            M[
                modelCHOL.Mu(1, i) @ M[sc],
                0,
            ],
            Mu[I[:, :, i]],
            clrmapMAN_states[I[i, :]],
            0.3,
        )

    ylabel("$\mathbf{K}^p$", "Interpreter", "Latex", "Fontsize", 10)
    xlabel("$t$", "Fontsize", 10, "Interpreter", "Latex")
    xlim(M[[-27, (xIn[I[end]] @ M[sc]) + 10]])
    ylim(M[[-27, 27]])
    set(gca, "FontSize", 10, "TickLabelInterpreter", "latex")
    if _not(LOAD_TOY_EX_DATA):
        _in = M[1:3]
        out = M[4:6]
        tic
        for j in reprosCHOL:
            tmpModel[I[j]] = GMR(
                model_r,
                M[
                    time(M[1 : (nbData - rdp)]),
                    dD(j).avgFe[I[:, (rdp + 1) : end]],
                ],
                _in,
                out,
            )

        elapsed_t_CHOL[I[2]] = mrdivide(toc, length(reprosCHOL))
    figure(handel_GMM_GMR_CHOL)
    subplot(3, 2, 5)
    hold("on")
    title("\fontsize{10}Desired Stiffness profile")
    for j in reprosCHOL:
        for i in round(linspace(1, nbData - rdp, 50)):
            Mu[I[:, :, i]] = zeros(2, 2)
            Mu[I[1, 1, i]] = tmpModel(j).Mu(1, i)
            Mu[I[1, 2, i]] = tmpModel(j).Mu(2, i)
            Mu[I[2, 2, i]] = tmpModel(j).Mu(3, i)
            Mu[I[:, :, i]] = (Mu[I[:, :, i]].H) @ Mu[I[:, :, i]]
            plotGMM2(
                M[
                    DataK(1, i) @ M[sc],
                    0,
                ],
                Mu[I[:, :, i]],
                M[[0, 1, 0]],
                0.3,
            )

    for j in reprosCHOL:
        l = 1
        for i in round(linspace(3, nbData - rdp, 5)):
            Mu[I[:, :, i]] = dD(j).KP[I[:, :, i]]
            plotGMM2(
                M[
                    DataK(1, i) @ M[sc],
                    0,
                ],
                Mu[I[:, :, i]],
                M[[1, 1, 0.5]],
                0.5,
            )
            l = l + 1

    ylabel("$\mathbf{\hat{K}}^p$", "Fontsize", 10, "Interpreter", "Latex")
    xlabel("$t$", "Fontsize", 10, "Interpreter", "Latex")
    axis(M[[-25, (DataK[I[1, end]] @ M[sc]) + 10, -25, 25]])
    set(gca, "FontSize", 10, "TickLabelInterpreter", "latex")
    subplot(3, 2, 6)
    hold("on")
    title("\fontsize{10}Influence of GMM components")
    for i in M[1 : (modelPD.nbStates)]:
        plot(
            xIn * 50,
            tmpModel(j).H[I[i, :]],
            "linewidth",
            2,
            "color",
            clrmapMAN_states[I[i, :]],
        )

    axis(M[[xIn(1), xIn[I[end]] * 50, 0, 1.02]])
    xlabel("$t$", "Fontsize", 10, "Interpreter", "Latex")
    ylabel("$h_k$", "Fontsize", 10, "Interpreter", "Latex")
    set(gca, "FontSize", 10, "TickLabelInterpreter", "latex")
    print(handel_GMM_GMR_CHOL, "-depsc2", "-r300", M["fig_GMM_GMR_CHOL"])
    figure(reprpduction)
    for j in 6:
        m1 = 1
        dt = 0.02
        xr = M[[]]
        xR1 = copy(g)
        x1[I[:, 1]] = copy(xR1)
        dx1[I[:, 1]] = zeros(2, 1)
        ddx1[I[:, 1]] = zeros(2, 1)
        f1 = zeros(2, 1)
        Fe = dD(j).avgFe
        for i in M[2 : (nbData - rdp)]:
            Mu[I[:, :, i]] = zeros(2, 2)
            Mu[I[1, 1, i]] = tmpModel(j).Mu(1, i)
            Mu[I[1, 2, i]] = tmpModel(j).Mu(2, i)
            Mu[I[2, 2, i]] = tmpModel(j).Mu(3, i)
            k1 = (Mu[I[:, :, i]].H) @ Mu[I[:, :, i]]
            f1[I[:, i - 1]] = M[k1] @ (x1[I[:, i - 1]] - xR1)
            ddx1[I[:, i]] = mrdivide(
                (Fe[I[:, i - 1]] - f1[I[:, i - 1]]) - (M[D] @ dx1[I[:, i - 1]]), m1
            )
            dx1[I[:, i]] = dx1[I[:, i - 1]] + (ddx1[I[:, i]] @ M[dt])
            x1[I[:, i]] = x1[I[:, i - 1]] + (dx1[I[:, i]] @ M[dt])

        n = 1
        for i in round(linspace(2, nbData - 2, 30)):
            x1x[I[:, n]] = x1[I[:, i]]
            n = n + 1

        plot(x1x[I[1, :]], x1x[I[2, :]], "b--o")

    for j in reprosCHOL:
        if j == reprosCHOL(1):
            cnt = 1
            b = round(linspace(1, nbData - rdp, 10))
        for i in M[1 : (nbData - rdp)]:
            Kpt = triu(ones(2))
            Kpt[I[logical(Kpt)]] = tmpModel(j).Mu(M[1:3], i)
            K_l = (Kpt.H) @ Kpt
            dD(j).K_l[I[:, :, i]] = copy(K_l)
            K_o = dD(j).KO[I[:, :, i + rdp]]
            chol_error[I[i, j]] = cholesky_dist(K_o, K_l)
            Log_Euclidean[I[i, j]] = Log_Euclidean_dist(K_l, K_o)
            Affine_Invariant[I[i, j]] = Affine_invariant_dist(K_l, K_o)
            Log_Determinant[I[i, j]] = Log_Determinant_dist(K_l, K_o)

        chol_e[I[j]] = mean(chol_error[I[:, j]] ** 2)
        bar1_chol[I[cnt]] = chol_e(j)
        logE_e[I[j]] = mean(Log_Euclidean[I[:, j]] ** 2)
        bar2_chol[I[cnt]] = logE_e(j)
        AffI_e[I[j]] = mean(Affine_Invariant[I[:, j]] ** 2)
        bar3_chol[I[cnt]] = AffI_e(j)
        logD_e[I[j]] = mean(Log_Determinant[I[:, j]] ** 2)
        bar4_chol[I[cnt]] = logD_e(j)
        chol_s[I[j]] = std(chol_error[I[:, j]] ** 2)
        bar1_chol_s[I[cnt]] = chol_s(j)
        logE_s[I[j]] = std(Log_Euclidean[I[:, j]] ** 2)
        bar2_chol_s[I[cnt]] = logE_s(j)
        AffI_s[I[j]] = std(Affine_Invariant[I[:, j]] ** 2)
        bar3_chol_s[I[cnt]] = AffI_s(j)
        logD_s[I[j]] = std(Log_Determinant[I[:, j]] ** 2)
        bar4_chol_s[I[cnt]] = logD_s(j)
        cnt = cnt + 1

    if _not(LOAD_TOY_EX_DATA):
        covOrder4to2, covOrder2to4 = set_covOrder4to2(modelPD.nbVar)
        disp("Learning GMM2 (K ellipsoids)...")
        _in = M[1:3]
        out = M[4 : (modelPD.nbVar)]
        tic
        modelPD = spd_init_GMM_kbins(X, modelPD, modelPD.nbSamples)
        modelPD, GAMMA, L = spd_EM_GMM(X, modelPD, nbData - rdp, _in, out)
        elapsed_t_Man[I[1]] = copy(toc)
    handel_GMM_GMR_SPD = copy(figure)
    htt = subplot(4, 2, M[1:2])
    hold("on")
    t1 = title("\fontsize{10}GMM/GMR based on Riemannian manifold")
    set(htt(1), "position", M[[0.1300, 0.95, 0.7750, 0.0]])
    htt.Visible = "off"
    t1.Visible = "on"
    subplot(3, 2, 1)
    hold("on")
    for n in M[1 : (modelPD.nbSamples)]:
        for i in round(linspace(1, nbData - rdp, 10)):
            plotGMM2(
                M[
                    i,
                    0,
                ],
                X(M[4:5], M[4:5], i + ((M[n - 1]) @ (nbData - rdp))),
                clrmap_demos[I[n, :]],
                0.4,
            )

    xlim(M[[-27, (nbData - rdp) + 10]])
    ylim(M[[-27, 27]])
    ylabel("$\mathbf{\tilde{K}}^\mathcal{P}$", "Interpreter", "Latex", "Fontsize", 11)
    xlabel("$t$", "Fontsize", 10, "Interpreter", "Latex")
    set(gca, "FontSize", 10, "TickLabelInterpreter", "latex")
    subplot(3, 2, 2)
    hold("on")
    for i in M[1 : (modelPD.nbStates)]:
        h, _ = plotGMM2(
            M[
                0,
                0,
            ],
            modelPD.MuMan(out, out, i),
            clrmapMAN_states[I[i, :]],
            0.3,
        )
        hh[I[i]] = h(1)

    xlabel("$x$", "Fontsize", 11, "Interpreter", "Latex")
    ylabel("$y$", "Fontsize", 11, "Interpreter", "Latex")
    axis("equal")
    xlim(M[[-25, 25]])
    ylim(M[[-25, 25]])
    set(gca, "FontSize", 10, "TickLabelInterpreter", "latex")
    subplot(3, 2, M[[3, 4]])
    hold("on")
    sc = mrdivide(1, modelPD.dt)
    for i in M[1 : size(X, 3)]:
        plotGMM2(
            M[
                X(1, 1, i) @ M[sc],
                0,
            ],
            X(out, out, i),
            M[[0.6, 0.6, 0.6]],
            0.1,
        )

    for i in M[1 : (modelPD.nbStates)]:
        plotGMM2(
            M[
                modelPD.MuMan(1, 1, i) @ M[sc],
                0,
            ],
            modelPD.MuMan(out, out, i),
            clrmapMAN_states[I[i, :]],
            0.3,
        )

    xlim(M[[-27, (nbData - rdp) + 10]])
    ylim(M[[-27, 27]])
    ylabel("$\mathbf{\tilde{K}}^\mathcal{P}$", "Interpreter", "Latex", "Fontsize", 11)
    xlabel("$t$", "Fontsize", 10, "Interpreter", "Latex")
    set(gca, "FontSize", 10, "TickLabelInterpreter", "latex")
    if _not(LOAD_TOY_EX_DATA):
        disp("Regression...")
        xInSPD = zeros(3, 3, nbData - rdp)
        xInSPD[I[1, 1, :]] = reshape(repmat(xIn, 1, 1), 1, 1, nbData - rdp)
        tic
        for j in reprosMan:
            for i in M[1 : (nbData - rdp)]:
                xInSPD[I[2, 2, i]] = dD(j).avgFe(1, i + rdp)
                xInSPD[I[3, 3, i]] = dD(j).avgFe(2, i + rdp)

            newModelPD[I[j]] = spd_GMR(modelPD, xInSPD, _in, out)

        elapsed_t_Man[I[2]] = mrdivide(toc, length(reprosMan))
    figure(handel_GMM_GMR_SPD)
    subplot(3, 2, 5)
    hold("on")
    for j in 6:
        for i in round(linspace(1, nbData - rdp, 50)):
            plotGMM2(
                M[
                    xInSPD(1, 1, i) @ M[sc],
                    0,
                ],
                newModelPD(j).Mu[I[:, :, i]],
                M[[0.2, 0.8, 0.2]],
                0.4,
            )

    set(gca, "FontSize", 8)
    ylabel("$\mathbf{\hat{K}}^\mathcal{P}$", "Fontsize", 11, "Interpreter", "Latex")
    xlabel("$t$", "Fontsize", 11, "Interpreter", "Latex")
    axis(M[[-25, (DataK[I[1, end]] @ M[sc]) + 10, -25, 25]])
    for j in 6:
        for i in round(linspace(3, nbData - rdp, 5)):
            Mu[I[:, :, i]] = dD(j).KP[I[:, :, i]]
            plotGMM2(
                M[
                    DataK(1, i) @ M[sc],
                    0,
                ],
                Mu[I[:, :, i]],
                M[[0.8, 0.2, 0.2]],
                0.4,
            )

    subplot(3, 2, 6)
    hold("on")
    for i in M[1 : (modelPD.nbStates)]:
        plot(
            xIn * 50,
            newModelPD(j).H[I[i, :]],
            "linewidth",
            2,
            "color",
            clrmapMAN_states[I[i, :]],
        )

    axis(M[[0, 100, 0, 1.02]])
    set(gca, "FontSize", 8)
    xlabel("$t$", "Fontsize", 11, "Interpreter", "Latex")
    ylabel("$h_k$", "Fontsize", 11, "Interpreter", "Latex")
    print(handel_GMM_GMR_SPD, "-depsc2", "-r300", M["fig_GMM_GMR_SPD"])
    print(handel_GMM_GMR_SPD, "-dpng", "-r300", M["fig_GMM_GMR_SPD"])
    figure(reprpduction)
    for j in 6:
        m1 = 1
        dt = 0.02
        xr = M[[]]
        xR1 = copy(g)
        x1[I[:, 1]] = copy(xR1)
        dx1[I[:, 1]] = zeros(2, 1)
        ddx1[I[:, 1]] = zeros(2, 1)
        f1 = zeros(2, 1)
        Fe = dD(j).avgFe
        for i in M[2 : (nbData - rdp)]:
            k1 = newModelPD(j).Mu[I[:, :, i]]
            f1[I[:, i - 1]] = M[k1] @ (x1[I[:, i - 1]] - xR1)
            ddx1[I[:, i]] = mrdivide(
                (Fe[I[:, i - 1]] - f1[I[:, i - 1]]) - (M[D] @ dx1[I[:, i - 1]]), m1
            )
            dx1[I[:, i]] = dx1[I[:, i - 1]] + (ddx1[I[:, i]] @ M[dt])
            x1[I[:, i]] = x1[I[:, i - 1]] + (dx1[I[:, i]] @ M[dt])

        n = 1
        for i in round(linspace(2, nbData - 2, 30)):
            x1x[I[:, n]] = x1[I[:, i]]
            n = n + 1

        plot(x1x[I[1, :]], x1x[I[2, :]], "--rs", "MarkerSize", 10)

    print(reprpduction, "-depsc2", "-r300", M["fig_reproduction"])
    for j in reprosMan:
        if j == reprosMan(1):
            cnt = 1
            b = round(linspace(1, nbData - rdp, 10))
        for i in M[1 : (nbData - rdp)]:
            K_l = newModelPD(j).Mu[I[:, :, i]]
            dD(j).K_l[I[:, :, i]] = copy(K_l)
            K_o = dD(j).KO[I[:, :, i + rdp]]
            chol_error[I[i, j]] = cholesky_dist(K_o, K_l)
            Log_Euclidean[I[i, j]] = Log_Euclidean_dist(K_l, K_o)
            Affine_Invariant[I[i, j]] = Affine_invariant_dist(K_l, K_o)
            Log_Determinant[I[i, j]] = Log_Determinant_dist(K_l, K_o)

        chol_e[I[j]] = mean(chol_error[I[:, j]] ** 2)
        bar1_man[I[cnt]] = chol_e(j)
        logE_e[I[j]] = mean(Log_Euclidean[I[:, j]] ** 2)
        bar2_man[I[cnt]] = logE_e(j)
        AffI_e[I[j]] = mean(Affine_Invariant[I[:, j]] ** 2)
        bar3_man[I[cnt]] = AffI_e(j)
        logD_e[I[j]] = mean(Log_Determinant[I[:, j]] ** 2)
        bar4_man[I[cnt]] = logD_e(j)
        chol_s[I[j]] = std(chol_error[I[:, j]] ** 2)
        bar1_man_s[I[cnt]] = chol_s(j)
        logE_s[I[j]] = std(Log_Euclidean[I[:, j]] ** 2)
        bar2_man_s[I[cnt]] = logE_s(j)
        AffI_s[I[j]] = std(Affine_Invariant[I[:, j]] ** 2)
        bar3_man_s[I[cnt]] = AffI_s(j)
        logD_s[I[j]] = std(Log_Determinant[I[:, j]] ** 2)
        bar4_man_s[I[cnt]] = logD_s(j)
        cnt = cnt + 1

    colorss = lines(modelCHOL.nbStates)
    model_series = M[
        [],
        [mean(bar2_chol), mean(bar2_man)],
        [mean(bar3_chol), mean(bar3_man)],
        [mean(bar4_chol), mean(bar4_man)],
    ]
    model_error = M[
        [],
        [std(bar2_chol), std(bar2_man)],
        [std(bar3_chol), std(bar3_man)],
        [std(bar4_chol), std(bar4_man)],
    ]
    handel_bar = figure("position", M[[1000, 500, 450, 110]])
    ax = copy(axes)
    h = bar(model_series, "BarWidth", 1)
    h(1).FaceColor = colorss[I[1, :]]
    h(2).FaceColor = colorss[I[2, :]]
    ax.YGrid = "on"
    ax.GridLineStyle = "-"
    xticks(ax, M[[1, 2, 3]])
    xticklabels(
        ax,
        C[
            "Log Euclidean",
            "Affine invariant",
            "Log Determinant",
        ],
    )
    ylabel("dist. error", "Fontsize", 12, "Interpreter", "Latex")
    lg = legend("GMR on Chol.", "GMR on $\bf{\mathcal{S}}_+^{m}$", "AutoUpdate", "off")
    set(lg, "Fontsize", 12, "Interpreter", "Latex")
    hold("on")
    ngroups = size(model_series, 1)
    nbars = size(model_series, 2)
    groupwidth = min(0.8, mrdivide(nbars, nbars + 1.5))
    for i in M[1:nbars]:
        x = ((M[1:ngroups]) - (groupwidth / 2)) + (
            mrdivide(((2 * i) - 1) @ M[groupwidth], 2 * nbars)
        )
        errorbar(
            x, model_series[I[:, i]], model_error[I[:, i]], "k", "linestyle", "none"
        )

    handel_bar.CurrentAxes.TickLabelInterpreter = "latex"
    ylim(M[[0, max(max(model_series)) + (2 * max(max(model_error)))]])
    set(ax, "Fontsize", 12)


if __name__ == "__main__":
    main()
