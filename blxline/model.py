"""Functions copied verbatim from UTime/GradientBoosting/__init__.py (github.com/Ambreghisalberti/UTime, commit bb6a5a0)
(only the functions used by the pipeline, plus the functions they call)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import random as rd


def temporal_split(data, columns, label_columns=None, test_size=0.2,
                   resolution=np.timedelta64(5,'s'), **kwargs):
    data['date'] = data.index.values
    data = pd.DataFrame(data.values, columns=data.columns.values)

    if label_columns is None:
        label_columns = ['label_BL']
    timestrain, timestest = [], []
    months = np.array(list(pd.date_range(start=data.date.values[0], end=data.date.values[-1],
                                         freq=kwargs.get('freq_split', timedelta(days=30)))) + [
                          pd.to_datetime(data.date.values[-1] + resolution)])
    for i in range(len(months) - 1):
        temp = data[np.logical_and(data.date.values >= months[i], data.date.values < months[i + 1])].index.values
        # The goal is to not take the last point, as it will also be part of the next month
        len_test = int(len(temp) * test_size)
        test = False
        count = 0
        while not test and count < 10:
            indice = rd.randint(0, len(temp) - len_test)
            temp_test = list(temp[indice:indice + len_test])
            timestest += temp_test
            temp_train = list(temp[:indice]) + list(temp[indice + len_test:])
            timestrain += temp_train
            test = len(temp_train) + len(temp_test) == len(temp)
            # Dataset should be split between train and test but points are left out for month {i}.
            count += 1
        if count == 10:
            raise Exception(f"Dataset should be split between train and test but after 10 tries, "
                            f"{len(temp) - (len(temp_train) + len(temp_test))}/{len(temp)} points are "
                            f"left out for month {i}.")
        """
        if len(temp_test) > 0:
            assert len(data[temp_test[0]:temp_test[-1]]) == len(temp_test), \
                ("The monthly testset portion is missing values from the dataset (it has more holes than "
                 "the original dataset)")
        """
    timestrain, timestest = np.array(timestrain), np.array(timestest)

    dftrain = data.loc[timestrain]  # Here what happens when the same date is here twice? Should give number indices instead of dates
    dftest = data.loc[timestest]

    dftrain = pd.DataFrame(dftrain.values, index=dftrain.date.values, columns=dftrain.columns.values).drop(
        columns=['date'])
    dftest = pd.DataFrame(dftest.values, index=dftest.date.values, columns=dftest.columns.values).drop(columns=['date'])

    if 'sat' in dftrain.columns:
        for sat in np.unique(dftrain.sat.values):
            subtrain = dftrain[dftrain.sat.values == sat]
            subtest = dftest[dftest.sat.values == sat]
            assert len(
                subtrain[subtrain.index.isin(subtest.index)]) == 0, \
                f"Trainset and testset should not have any point in common for {sat}!"
    else:
        assert len(dftrain[dftrain.index.isin(dftest.index)]) == 0, \
            "Trainset and testset should not have any point in common!"

    assert len(dftrain) + len(dftest) == len(data), (f"Dataset should be split between train and test but "
                                                     f"{len(data) - (len(dftrain) + len(dftest))}/{len(data)} "
                                                     f"points are left out.")

    return (dftrain.loc[:, columns].values, dftest.loc[:, columns].values,
            dftrain.loc[:, label_columns].values, dftest.loc[:, label_columns].values,
            timestrain, timestest)


def compute_scores(pred, truth, threshold=0.5):
    pred = (pred > threshold).astype(int)
    TP = (pred * truth).sum()
    FP = (pred * (1 - truth)).sum()
    TN = ((1 - pred) * (1-truth)).sum()
    FN = ((1 - pred) * truth).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return TP, FP, TN, FN, precision, recall


def recall_precision_curve(pred_proba, label, **kwargs):
    if 'pred' in kwargs:
        pred_proba = kwargs['pred']
    elif 'model' in kwargs and 'xtest' in kwargs:
        pred_proba = kwargs['model'].predict_proba(kwargs['xtest'])[:, 1]
    label = label.flatten()

    precisions, recalls = [], []
    for threshold in np.linspace(0, 1, 100):
        _, _, _, _, precision, recall = compute_scores(pred_proba, label, threshold=threshold)
        precisions.append(precision)
        recalls.append(recall)

    if 'ax' not in kwargs:
        _, ax = plt.subplots()
    else:
        ax = kwargs['ax']
    ax.scatter(recalls, precisions)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.axhline(label.sum() / len(label), linestyle='--', color='grey', alpha=0.5)
    return recalls, precisions


def find_best_threshold(pred_proba, label, **kwargs):
    thresholds = np.linspace(0, 1, 100)
    recalls, precisions = recall_precision_curve(pred_proba, label, **kwargs)
    i = np.nanargmax(np.array(precisions)*np.array(recalls)).item()
    return thresholds[i]


def get_all_feature_choices(features_yes_or_no, features_multiple_choice):
    all_combinaisons = [[]]
    for _ in features_yes_or_no:
        all_combinaisons = ([comb + [0] for comb in all_combinaisons] +
                            [comb + [1] for comb in all_combinaisons])

    for choices in features_multiple_choice:
        nb_choices = len(choices)
        all_new_combinaisons = []
        for i in range(nb_choices):
            all_new_combinaisons += [comb + [i] for comb in all_combinaisons]
        all_combinaisons = all_new_combinaisons

    return all_combinaisons


def get_all_feature_combinaisons_multiple_choices(features_yes_or_no, features_multiple_choice, features_mandatory):
    all_combinaisons = np.array(get_all_feature_choices(features_yes_or_no, features_multiple_choice))

    combinaisons_features = []
    for comb in all_combinaisons:
        comb_features = features_mandatory.copy()
        for i, features in enumerate(features_yes_or_no):
            if comb[i] == 1:
                comb_features += features
        for i, choices in enumerate(features_multiple_choice):
            choice = comb[len(features_yes_or_no) + i]
            comb_features += choices[choice]
        combinaisons_features += [comb_features]
    return combinaisons_features


def get_name_choice(choice):
    temp = str(choice)[1:-1].split(', ')
    temp2 = ''
    for t in temp:
        temp2 += t
    return temp2


def add_scores(ytest, precisions, recalls, threshold=0.5, **kwargs):
    if 'pred' in kwargs:
        pred = kwargs['pred']
    elif ('model' in kwargs) and ('xtest' in kwargs):
        pred = kwargs['model'].predict_proba(kwargs['xtest'])[:,1]
    pred = (pred > threshold).astype(int)
    ytest = ytest.flatten()
    TP, FP, TN, FN, precision, recall = compute_scores(pred, ytest, threshold=threshold)
    precisions.append(precision)
    recalls.append(recall)
    return precisions, recalls


def x_y_set(df, scaler, columns):
    y = df.label_BL.values
    x = df[columns]
    x.loc[:, :] = scaler.transform(x.values)
    return x, y


def get_all_features_ensemble_model(model_results):
    all_features = []
    for features in model_results['all_features']:
        all_features += features
    all_features = list(np.unique(np.array(all_features)).astype('str'))
    return all_features


def pred_ensemble(x, model_results):
    pred = np.zeros(len(x))
    for model, scaler, features in zip(model_results['models'], model_results['scalers'],
                                       model_results['all_features']):
        pred += model.predict_proba(scaler.transform(x[features]))[:, 1]
    pred = pred / len(model_results['models'])
    return pred


def get_features_from_choice(choice, mandatory, features_yes_or_no, features_multiple_choice):
    features = list(np.array(mandatory).copy())
    for i,c in enumerate(choice):
        if i < len(features_yes_or_no):
            if c == '1':
                features += features_yes_or_no[i]
        else:
            c = int(c)
            features += features_multiple_choice[i-len(features_yes_or_no)][c]
    return features


def choose_best_features():
    # Feature groups
    mandatory = ['Bn_MP', 'Btan1_MP', 'Btan2_MP', 'max_coldions']  # 1
    # mandatory = []

    V_features = ['Vx', 'Vy', 'Vz']  # OR
    V_features2 = ['Vtan1_MP', 'Vtan2_MP', 'Vn_MP']

    norms = ['V', 'B']  # 0 or 1
    jets = ['gap_to_MSH_Vtan1_MP_over_Va', 'gap_to_MSH_Vtan2_MP_over_Va']  # 0 or 1

    particle = ['Np', 'Tp', 'anisotropy']  # OR
    particle2 = ['Np', 'Tpara', 'Tperp']  # OR
    log_particle = ['logNp', 'logTp', 'anisotropy']  # OR
    log_particle2 = ['logNp', 'logTpara', 'logTperp']

    spectro_features = [f'spectro_{i}' for i in range(32)]  # OR
    peak_detection_features = ['energy_main_pop', 'width_main_pop', 'flux_main_pop']  # OR
    # gaussian_fit_features = ['max_msp_popu','center_msp_popu','max_msh_popu','center_msh_popu']

    cold_ions_more = ['center_coldions', 'std_coldions']  # 0 or 1

    gap_to_MSH_features = ['gap_to_MSH_Vx', 'gap_to_MSH_Vy', 'gap_to_MSH_Vz', 'gap_to_MSH_V', 'gap_to_MSH_logTp',
                           'gap_to_MSH_logNp', 'gap_to_MSH_anisotropy']  # OR
    ref_MSH_features = ['ref_MSH_Vx', 'ref_MSH_Vy', 'ref_MSH_Vz', 'ref_MSH_V', 'ref_MSH_logTp', 'ref_MSH_logNp',
                        'ref_MSH_anisotropy']  # OR
    relative_gap_MSH_features = [f'relative_gap_with_MSH_{feat}' for feat in
                                 ['Vx', 'Vy', 'Vz', 'V', 'Tp', 'Np', 'anisotropy']]

    gap_to_MSH_features2 = [f'gap_to_MSH_{feat}' for feat in ['Vtan1_MP', 'Vtan2_MP', 'Vn_MP']]  # OR
    ref_MSH_features2 = [f'ref_MSH_{feat}' for feat in ['Vtan1_MP', 'Vtan2_MP', 'Vn_MP']]  # OR
    relative_gap_MSH_features2 = [f'relative_gap_with_MSH_{feat}' for feat in ['Vtan1_MP', 'Vtan2_MP', 'Vn_MP']]

    individual_features = ['Bz_over_B']  # That don't fit in groups

    # Feature selection setup
    features_yes_or_no = [norms, jets, cold_ions_more, individual_features, gap_to_MSH_features2]
    features_multiple_choice = [
        [V_features, V_features2],
        [particle, particle2, log_particle, log_particle2],
        [spectro_features, peak_detection_features],  # gaussian_fit_features],
        [gap_to_MSH_features, ref_MSH_features, relative_gap_MSH_features]
    ]

    choices = get_all_feature_choices(features_yes_or_no, features_multiple_choice)
    features_combinaisons = get_all_feature_combinaisons_multiple_choices(features_yes_or_no, features_multiple_choice,
                                                                          mandatory)

    # Load performance data
    precisions = pd.read_pickle('/home/ghisalberti/GradientBoosting/diagnostics/all_combinaisons/all_precisions.pkl')
    recalls = pd.read_pickle('/home/ghisalberti/GradientBoosting/diagnostics/all_combinaisons/all_recalls.pkl')

    # Filter best-performing combinations
    all_features = []
    for choice in np.array(list(precisions.keys()))[
        np.logical_and(np.array(list(precisions.values())) > 0.83, np.array(list(recalls.values())) > 0.821)]:
        # for choice in np.array(list(precisions.keys()))[np.array(list(precisions.values()))*np.array(list(recalls.values())) > 0.833**2]:
        features = get_features_from_choice(choice, mandatory, features_yes_or_no, features_multiple_choice)
        all_features.append(features)

    # Get unique features across combinations
    all_features_unique = []
    for features in all_features:
        all_features_unique += list(features)
    all_features_unique = np.unique(np.array(all_features_unique + ['sat', 'label_BL']).flatten())

    return all_features


