import mat2py as mp
from mat2py.core import *


def main():
    model.nbStates = 4
    model.nbFrames = 3
    model.nbVar = 3
    model.kP = 600
    model.kV = 50
    model.dt = 0.01
    needsModel = 0
    BICselection = 0
    stiffEstimate = 0
    saveModelMat = 1
    saveStiffEst = 1
    dataPath = "data/"
    load(M[[dataPath, "Data.mat"]])
    nbData = size(s(1).DataP, 2)
    demons = M[1:5]
    nbSamples = length(demons)
    posId = M[1:2]
    velId = M[3:4]
    accId = M[5:6]
    for n in demons:
        s(n).nbData = copy(nbData)
        s(n).Data = zeros(model.nbVar, s(n).nbData)
        s(n).Data = M[
            (M[M[1:nbData]]) @ (model.dt),
            (
                (
                    (s(n).DataP[I[accId, :]] @ (M[mrdivide(1, model.kP)]))
                    + (M[s(n).DataP[I[velId, :]]] @ (mrdivide(model.kV, model.kP)))
                )
                - ((s(n).DataF) * 0.0)
            )
            + s(n).DataP[I[posId, :]],
        ]

    Data = M[[]]
    cnt = 1
    for n in demons:
        for m in M[1 : (model.nbFrames)]:
            tt1 = 1 + (M[nbData] @ (cnt - 1))
            tt2 = M[cnt] @ nbData
            Data[I[:, m, tt1:tt2]] = mldivide(
                s(n).p(m).A, (s(n).Data) - repmat(s(n).p(m).b, 1, s(n).nbData)
            )

        cnt = cnt + 1

    fprintf("Parameters estimation of TP-GMM:\n")
    if needsModel:
        if BICselection:
            fprintf("BIC-based model selection...\n")
            K = M[1:10]
            BICvalues = M[[]]
            penalty = M[[]]
            LogL = M[[]]
            m = M[[]]
            for k in K:
                model.nbStates = copy(k)
                model = init_tensorGMM_timeBased(Data, model)
                model, _ = EM_tensorGMM(Data, model)
                m(k).model = copy(model)
                for n in demons:
                    Mu, Sigma = gaussianProduct(model, s(n).p)
                    for i in M[1 : (model.nbStates)]:
                        Lklhd[I[i, :]] = M[model.Priors(i)] @ gaussPDF(
                            s(n).Data, Mu[I[:, i]], Sigma[I[:, :, i]]
                        )

                    LL[I[n]] = sum(log(sum(Lklhd, 1)))
                    auxBIC[I[n]], auxPenalty[I[n]] = BIC(
                        s(n).Data, model, LL(n), nbData
                    )

                LogL = M[[LogL, mean(LL)]]
                BICvalues = M[[BICvalues, mean(auxBIC)]]
                penalty = M[[penalty, mean(auxPenalty)]]

            _, minInd = min(BICvalues)
            model = m(K(minInd)).model
            fprintf("Done!\n")
        else:
            model = init_tensorGMM_timeBased(Data, model)
            model = EM_tensorGMM(Data, model)
        if saveModelMat:
            save(M[[dataPath, "model.mat"]], "model")
    else:
        fprintf("Loading model...")
        load(M[[dataPath, "model.mat"]])
        fprintf("Done!\n")
    if stiffEstimate:
        fprintf("Stiffness matrices estimation...")
        Y = M[[]]
        W = M[[]]
        X = M[[]]
        for n in demons:
            w = M[[]]
            auxX = M[[]]
            Mu, Sigma = gaussianProduct(model, s(n).p)
            for i in M[1 : (model.nbStates)]:
                w[I[:, i]] = gaussPDF(s(n).Data[I[1, :]], Mu(1, i), Sigma(1, 1, i))
                auxX[I[:, :, i]] = (
                    repmat(Mu(M[2:3], i), 1, nbData) - s(n).DataP[I[posId, :]]
                )

            auxW = w / repmat(sum(w, 2), 1, model.nbStates)
            auxY = (
                s(n).DataP[I[accId, :]] + ((M[model.kV]) @ s(n).DataP[I[velId, :]])
            ) - (s(n).DataF)
            W = M[
                W,
                auxW,
            ]
            X = M[[X, auxX]]
            Y = M[[Y, auxY]]

        for i in M[1 : (model.nbStates)]:
            XtW = ((X[I[:, :, i]].H) * repmat(W[I[:, i]], 1, size(X, 1))).H
            YtW = Y * repmat(W[I[:, i]].H, size(Y, 1), 1)
            d = length(posId)
            cvx_begin("sdp")
            variable("Kcvx(d,d)", "symmetric")
            minimize(norm((Kcvx @ M[XtW]) - YtW, 2))
            Kcvx >= eye(d)
            cvx_end
            KpCVX[I[:, :, i]] = copy(Kcvx)

        if saveStiffEst:
            save(M[[dataPath, "KpCVX.mat"]], "KpCVX")
        fprintf("Done!\n")
    else:
        fprintf("Loading stiffness matrices...")
        load(M[[dataPath, "KpCVX.mat"]])
        fprintf("Done!\n")
    KpMax = 1800
    KpMin = 400
    for k in M[1 : (model.nbStates)]:
        Kp[I[:, :, k]] = diag(diag(KpCVX[I[:, :, k]]))
        V[I[:, :, k]], Dtmp1 = eig(Kp[I[:, :, k]])
        _lambda[I[:, k]] = diag(Dtmp1)

    lambda_min = min(min(_lambda))
    lambda_max = max(max(_lambda))
    for k in M[1 : (model.nbStates)]:
        lambdaFactor = (_lambda[I[:, k]] - lambda_min) / (lambda_max - lambda_min)
        Dtmp1 = diag(((KpMax - KpMin) * lambdaFactor) + KpMin)
        Kp[I[:, :, k]] = mrdivide(V[I[:, :, k]] @ M[Dtmp1], V[I[:, :, k]])

    repros = M[M[6:8]]
    _in = 1
    out = M[[2, 3]]
    model.estKp = copy(Kp)
    for n in repros:
        x0 = s(n).DataP(posId, 1)
        input = (M[M[1:nbData]]) @ (model.dt)
        r[I[n]] = reproduction_tensorGMM_DS_Kp(
            input, model, s(n), x0, _in, out, s(n).DataF
        )

    if needsModel and BICselection:
        figure("position", M[[20, 50, 1100, 600]], "color", M[[1, 1, 1]])
        subplot(2, 2, 1)
        hold("on")
        box("on")
        plot(BICvalues, "-b", "LineWidth", 2)
        plot(BICvalues, "or", "MarkerSize", 8, "MarkerFaceColor", "r")
        set(gca, "XTick", M[1 : length(K)], "XTickLabel", K)
        set(gca, "FontSize", 22)
        xlabel("$K$", "Fontsize", 18, "interpreter", "latex")
        ylabel("BIC", "Fontsize", 18, "interpreter", "latex")
    figure(
        "PaperPosition",
        M[[0, 0, 18, 4.5]],
        "position",
        M[[100, 100, 1400, 900]],
        "color",
        M[[1, 1, 1]],
    )
    GMMclrs = M[
        M[[0.95, 0.10, 0.10]],
        M[[0.12, 0.12, 0.99]],
        M[[0.95, 0.56, 0.12]],
        M[[0.15, 0.95, 0.45]],
    ]
    xLbls = M[
        "$x_1$",
        "$x_2$",
    ]
    fLbls = M[
        "$f_1$",
        "$f_2$",
    ]
    ySLbls = M[
        "$y_1^{\mathcal{S}}$",
        "$y_2^{\mathcal{S}}$",
    ]
    yTLbls = M[
        "$y_1^{\mathcal{T}}$",
        "$y_2^{\mathcal{T}}$",
    ]
    yILbls = M[
        "$y_1^{\mathcal{I}}$",
        "$y_2^{\mathcal{I}}$",
    ]
    frmLbls(1).Lbls = copy(ySLbls)
    frmLbls(2).Lbls = copy(yTLbls)
    frmLbls(3).Lbls = copy(yILbls)
    xLims[I[:, :, 1]] = M[
        M[[-0.05, 2.0]],
        M[[-0.50, 2.0]],
    ]
    xLims[I[:, :, 2]] = M[
        M[[-2.0, 0.5]],
        M[[-2.0, 0.5]],
    ]
    xLims[I[:, :, 3]] = M[
        M[[-1.0, 1.5]],
        M[[-1.5, 1.0]],
    ]
    subplot(2, 4, 1)
    hold("on")
    box("on")
    for n in demons:
        for m in M[1 : (model.nbFrames)]:
            plot2Dframe(
                s(n).p(m, 1).A(2, M[2:3]),
                s(n).p(m, 1).A(3, M[2:3]),
                s(n).p(m, 1).b(M[2:3]),
                GMMclrs[I[m, :]],
                GMMclrs[I[m, :]],
                0.2,
            )

        plot(s(n).DataP[I[1, :]], s(n).DataP[I[2, :]], "-", "color", M[[0.3, 0.3, 0.3]])

    xlabel("$x_1$", "Fontsize", 18, "interpreter", "latex")
    ylabel("$x_2$", "Fontsize", 18, "interpreter", "latex")
    axis("equal")
    axis("square")
    set(gca, "Fontsize", 16)
    set(gca, "Layer", "top")
    axis(M[[-1.0, 1.0, -1.0, 1.0]])
    text(-0.7, 1.25, "Demonstrations", "Fontsize", 18, "interpreter", "latex")
    for m in M[1 : (model.nbFrames)]:
        cnt = 1
        for n in M[1:nbSamples]:
            tt1 = 1 + (M[nbData] @ (cnt - 1))
            tt2 = M[cnt] @ nbData
            subplot(2, 4, m + 1)
            hold("on")
            box("on")
            if n == 1:
                plotGMM(
                    reshape(model.Mu[I[2:3, m, :]], M[[2, model.nbStates]]),
                    reshape(model.Sigma[I[2:3, 2:3, m, :]], M[[2, 2, model.nbStates]]),
                    GMMclrs[I[m, :]],
                )
                plot2Dframe(
                    M[[1, 0]],
                    M[[0, 1]],
                    zeros(1, 2),
                    GMMclrs[I[m, :]],
                    GMMclrs[I[m, :]],
                    0.2,
                )
            plot(
                reshape(Data(2, m, M[tt1:tt2]), M[[1, nbData]]),
                reshape(Data(3, m, M[tt1:tt2]), M[[1, nbData]]),
                "-",
                "color",
                M[[0.3, 0.3, 0.3]],
            )
            plot(
                Data(2, m, tt1),
                Data(3, m, tt1),
                ".",
                "Markersize",
                18,
                "color",
                M[[0.0, 0.0, 0.0]],
            )
            plot(
                Data(2, m, tt2),
                Data(3, m, tt2),
                "x",
                "Markersize",
                10,
                "color",
                M[[0.0, 0.0, 0.0]],
            )
            xlabel(frmLbls(m).Lbls[I[1, :]], "Fontsize", 18, "interpreter", "latex")
            ylabel(frmLbls(m).Lbls[I[2, :]], "Fontsize", 18, "interpreter", "latex")
            axis("equal")
            axis("square")
            set(gca, "Fontsize", 16)
            set(gca, "Layer", "top")
            axis(M[[xLims[I[1, :, m]], xLims[I[2, :, m]]]])
            cnt = cnt + 1

    text(
        -5.2,
        1.3,
        "Attractor projected on the task parameters",
        "Fontsize",
        18,
        "interpreter",
        "latex",
    )
    cnt = 1
    for n in repros:
        subplot(2, 3, cnt + 3)
        hold("on")
        box("on")
        plotGMM(r(n).Mu[I[2:3, :]], r(n).Sigma[I[2:3, 2:3, :]], GMMclrs[I[4, :]])
        plot2Dframe(
            M[[1, 0]], M[[0, 1]], zeros(1, 2), GMMclrs[I[4, :]], GMMclrs[I[4, :]], 0.2
        )
        plot2Dframe(
            M[[1, 0]],
            M[[0, 1]],
            s(n).p(1).b(M[2:3]),
            GMMclrs[I[1, :]],
            GMMclrs[I[1, :]],
            0.2,
        )
        plot2Dframe(
            M[[1, 0]],
            M[[0, 1]],
            s(n).p(2).b(M[2:3]),
            GMMclrs[I[2, :]],
            GMMclrs[I[2, :]],
            0.2,
        )
        h1 = plot(
            r(n).Data[I[8, :]], r(n).Data[I[9, :]], "--", "color", M[[0.7, 0.7, 0.7]]
        )
        for t in M[1:nbData]:
            h2 = quiver(
                r(n).Data(2, t),
                r(n).Data(3, t),
                s(n).DataF(1, t),
                s(n).DataF(2, t),
                0.02,
                "color",
                M[[0.8, 0.8, 0.6]],
            )

        h3 = plot(
            r(n).Data[I[2, :]], r(n).Data[I[3, :]], "-", "color", M[[0.3, 0.3, 0.3]]
        )
        plot(
            r(n).Data(2, 1),
            r(n).Data(3, 1),
            ".",
            "Markersize",
            18,
            "color",
            M[[0.0, 0.0, 0.0]],
        )
        plot(
            r(n).Data[I[2, end]],
            r(n).Data[I[3, end]],
            "x",
            "Markersize",
            10,
            "color",
            M[[0.0, 0.0, 0.0]],
        )
        xlabel("$x_1$", "Fontsize", 18, "interpreter", "latex")
        ylabel("$x_2$", "Fontsize", 18, "interpreter", "latex")
        axis("equal")
        axis("square")
        set(gca, "Fontsize", 16)
        set(gca, "Layer", "top")
        axis(M[[-1.1, 1.1, -1.1, 1.1]])
        cnt = cnt + 1

    text(
        -4.2,
        1.2,
        "Reproductions for different task parameters",
        "Fontsize",
        18,
        "interpreter",
        "latex",
    )
    legend(
        M[[h1, h2, h3]],
        "Attractor",
        "Sensed force",
        "Robot pos",
        "Location",
        "southeast",
    )
    figure(
        "PaperPosition",
        M[[0, 0, 18, 4.5]],
        "position",
        M[[100, 100, 1500, 600]],
        "color",
        M[[1, 1, 1]],
    )
    xx = round(linspace(1, 64, model.nbStates))
    clrmap = colormap("Jet")
    clrmap = min(clrmap[I[xx, :]], 0.95)
    for i in M[1:2]:
        subplot(2, 3, i)
        hold("on")
        box("on")
        plot(
            r(n).Data[I[1, :]],
            r(n).Data[I[7 + i, :]],
            "--",
            "color",
            M[[0.7, 0.7, 0.7]],
        )
        plot(
            r(n).Data[I[1, :]], r(n).Data[I[1 + i, :]], "-", "color", M[[0.3, 0.3, 0.3]]
        )
        ylabel(xLbls[I[i, :]], "Fontsize", 18, "interpreter", "latex")
        set(gca, "Fontsize", 16)
        axis(M[[0.0, 1.0, -1.0, 1.0]])
        if i == 1:
            legend("Attractor", "Robot pos", "Location", "northwest")
        subplot(2, 3, i + 3)
        hold("on")
        box("on")
        plot(
            r(n).Data[I[1, :]], r(n).Data[I[9 + i, :]], "-", "color", M[[0.3, 0.3, 0.3]]
        )
        xlabel("$t$", "Fontsize", 18, "interpreter", "latex")
        ylabel(fLbls[I[i, :]], "Fontsize", 18, "interpreter", "latex")
        set(gca, "Fontsize", 16)
        axis(M[[0.0, 1.0, -7.0, 7.0]])

    text(
        -0.3,
        28,
        "Evolution over time of one reproduction of the task",
        "Fontsize",
        18,
        "interpreter",
        "latex",
    )
    subplot(2, 3, 3)
    hold("on")
    box("on")
    for i in M[1 : (model.nbStates)]:
        plot(input, r(repros(1)).H[I[i, :]], "color", clrmap[I[i, :]], "LineWidth", 4)

    axis(M[[0.02, 1, -0.15, 1.15]])
    ylim(M[[-0.15, 1.15]])
    ylabel("$\gamma$", "Fontsize", 20, "interpreter", "latex")
    set(gca, "Fontsize", 16)
    subplot(2, 3, 6)
    hold("on")
    box("on")
    plot(
        input,
        reshape(r(repros(1)).Kp[I[1, 1, :]], M[[1, size(r(repros(1)).Kp, 3)]]),
        "r-",
        "LineWidth",
        4,
    )
    plot(
        input,
        reshape(r(repros(1)).Kp[I[2, 2, :]], M[[1, size(r(repros(1)).Kp, 3)]]),
        "g--",
        "LineWidth",
        4,
    )
    axis(M[[0.02, 1, KpMin - 100, KpMax + 100]])
    xlabel("$t$", "Fontsize", 20, "interpreter", "latex")
    ylabel("$K^{\mathcal{P}}$", "Fontsize", 20, "interpreter", "latex")
    set(gca, "Fontsize", 16)
    legend("x_1", "x_2", "Location", "northwest")


if __name__ == "__main__":
    main()
