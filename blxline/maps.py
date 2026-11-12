"""Functions copied verbatim from swapp/make_plots/plot_maps.py (github.com/Ambreghisalberti/swapp, commit 23c5106)
(only the functions used by the pipeline, plus the functions they call)."""
from spok.coordinates.coordinates import cartesian_to_spherical, spherical_to_cartesian
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from spok.models.planetary import mp_shue1997
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from spok.models.planetary import Magnetosheath
import matplotlib.patches as mpatches
import pandas as pd
from datetime import timedelta
from skimage import measure


def gaussian_filter_nan_datas(df, sigma):
    if sigma > 0:
        # Get the coordinates of the non-NaN values
        nan_mask = np.isnan(df)
        x, y = np.indices(df.shape)
        valid_points = ~nan_mask
        points = np.vstack((x[valid_points], y[valid_points])).T
        values = df[valid_points]
        # Interpolate the NaN values based on the surrounding values linearly
        interpolated_arr = df.copy()
        interpolated_arr[nan_mask] = griddata(points, values, (x[nan_mask], y[nan_mask]), method='linear')
        assert np.isnan(interpolated_arr).sum() <= nan_mask.sum(), "The first interpolation step adds NaNs."

        # Red-do a step to also fill the ones that are not surrounded by values (nearest inseatd of linear)
        nan_mask2 = np.isnan(interpolated_arr)
        x, y = np.indices(interpolated_arr.shape)
        valid_points = ~nan_mask2
        points = np.vstack((x[valid_points], y[valid_points])).T
        values = interpolated_arr[valid_points]
        interpolated_arr[nan_mask2] = griddata(points, values, (x[nan_mask2], y[nan_mask2]), method='nearest')
        assert np.isnan(
            interpolated_arr).sum() == 0, "The Nanfilling steps have not workdes : there still are Nans in the array."

        # Apply the Gaussian filter to the interpolated array, and add again the initial nans
        filtered_arr = gaussian_filter(interpolated_arr, sigma=sigma)
        filtered_arr[nan_mask] = np.nan

        assert np.isnan(filtered_arr).sum() == nan_mask.sum(), (
            "The gaussian filter should not add ""or remove any NaN value.")
        return filtered_arr
    else:
        return df


def gaussian_filter_nan_1d(arr, sigma):
    if sigma > 0:
        # Create an array of indices
        x = np.arange(len(arr))
        nan_mask = np.isnan(arr)

        # If there are no NaNs or all NaNs, return as is
        if nan_mask.sum() == 0:
            return gaussian_filter(arr, sigma=sigma)
        if nan_mask.sum() == len(arr):
            return arr

        # Step 1: Interpolate NaN values linearly based on valid indices
        valid_idx = x[~nan_mask]
        valid_values = arr[~nan_mask]

        # Create an interpolation function
        # kind='linear' for the first pass, 'nearest' for edges
        f_linear = interp1d(valid_idx, valid_values, kind='linear',
                            bounds_error=False, fill_value="extrapolate")
        f_nearest = interp1d(valid_idx, valid_values, kind='nearest',
                             bounds_error=False, fill_value="extrapolate")

        interpolated_arr = arr.copy()
        # Fill linear gaps
        interpolated_arr[nan_mask] = f_linear(x[nan_mask])

        # Step 2: Fill remaining NaNs (edges) with nearest neighbor
        nan_mask2 = np.isnan(interpolated_arr)
        interpolated_arr[nan_mask2] = f_nearest(x[nan_mask2])

        # Final safety check
        assert np.isnan(interpolated_arr).sum() == 0, "NaNs remain after filling."

        # Step 3: Apply Gaussian filter and re-apply original NaN mask
        filtered_arr = gaussian_filter(interpolated_arr, sigma=sigma)
        filtered_arr[nan_mask] = np.nan

        return filtered_arr
    else:
        return arr


def reposition(df, **kwargs):
    if 'Rmin' in kwargs:
        rmin = kwargs['Rmin']
    else:
        rmin = df.r_MP_Shue.values / 2
    rmax = df.r_MP_Shue.values
    df['normalized_R'] = rmin + (rmax - rmin) * df.normalized_logNoverT.values
    norm_x, norm_y, norm_z = spherical_to_cartesian(df.normalized_R.values, df.theta.values, df.phi.values)
    df['normalized_X'] = norm_x
    df['normalized_Y'] = norm_y
    df['normalized_Z'] = norm_z
    return df


def make_mp_grid(**kwargs):
    N_grid = kwargs.get('N_grid', 300)
    coord = kwargs.get('coord', 'spherical')
    if coord == 'spherical':
        th = np.linspace(0, 0.5 * np.pi, N_grid)
        ph = np.linspace(-np.pi, np.pi, 2 * N_grid)
        theta, phi = np.meshgrid(th, ph)
        msh = Magnetosheath()
        Xmp, Ymp, Zmp = msh.magnetopause(theta, phi)
    elif coord == 'cartesian':
        th = np.linspace(0, 0.5 * np.pi, 3000)
        ph = np.linspace(-np.pi, np.pi, 3000)
        theta, phi = np.meshgrid(th, ph)
        msh = Magnetosheath()
        X, Y, Z = msh.magnetopause(theta, phi)
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        Ymp = np.linspace(-15, 15, 2*N_grid)
        Zmp = np.linspace(-15, 15, N_grid)
        Ymp, Zmp = np.meshgrid(Ymp, Zmp)
        Ymp, Zmp = Ymp.flatten(), Zmp.flatten()

        model = KNeighborsRegressor(n_neighbors=1, weights='distance', n_jobs=1)
        model.fit(np.array([Y, Z]).T, X)
        Xmp = model.predict(np.array([Ymp, Zmp]).T)
        Xmp = Xmp.reshape((N_grid, 2*N_grid))
        Ymp = Ymp.reshape((N_grid, 2*N_grid))
        Zmp = Zmp.reshape((N_grid, 2*N_grid))
    else:
        raise Exception(f'coord should be either spherical or cartesian but is {coord}.')

    return Xmp, Ymp, Zmp


def arc_patch(center, radius, theta1, theta2, ax=None, resolution=50, **kwargs):  # Adapted from stack overflow
    # make sure ax is not empty
    if ax is None:
        ax = plt.gca()
    # generate the points
    theta = np.linspace(theta1, theta2, resolution)
    points = np.vstack((list(radius*np.cos(theta) + center[0])+[center[0]],
                        list(radius*np.sin(theta) + center[1])+[center[1]]))
    # build the polygon and add it to the axes
    poly = mpatches.Polygon(points.T, closed=True, **kwargs)
    ax.add_patch(poly)
    return poly


def plot_CLA_sector(cx, cy, r, theta1, theta2, ax):
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color='k', alpha=0.5)
    arc_patch((cx, cy), r, (np.pi/2-theta2), (np.pi/2-theta1), ax=ax, fill=True, color='red', alpha=0.3)


def make_CLA_slice(df, feature, min_cla, max_cla):
    if (min_cla >= -np.pi) and (max_cla <= np.pi):
        temp = df[df[feature].values >= min_cla]
        temp = temp[temp[feature].values < max_cla]
    elif min_cla < -np.pi:
        temp1 = df[df[feature].values >= min_cla + 2 * np.pi]
        temp2 = df[df[feature].values < max_cla]
        temp = pd.concat([temp1, temp2])
    else:  # sectors_CLA[i+1] > np.pi
        temp1 = df[df[feature].values >= min_cla]
        temp2 = df[df[feature].values < max_cla - 2 * np.pi]
        temp = pd.concat([temp1, temp2])
    return temp


def make_slice(df, feature, min_val, max_val):
    if 'CLA' in feature or 'COA' in feature:
        return make_CLA_slice(df, feature, min_val, max_val)
    else:
        return df[np.logical_and(min_val < df[feature].values, df[feature].values < max_val)]


def associate_SW_Safrankova(X_sat, omni, BS_standoff, dtm=0, sampling_time='5S', vx_median=-406.2):
    if dtm != 0:
        vxmean = abs(omni.Vx.rolling(int((2*dtm+1)*timedelta(minutes=1)/(omni.index[-1]-omni.index[-2])),
                                     center=True, min_periods=1).mean())
    else:
        vxmean = abs(omni.Vx)
    BS_x0 = BS_standoff[BS_standoff.index.isin(X_sat.index)]
    BS_x0 = BS_x0.fillna(13.45)
    lag = np.array(np.round((BS_x0.values-X_sat.values)*6371/vx_median),dtype='timedelta64[s]')
    time = (X_sat.index-lag).round(sampling_time)
    vx = pd.Series(name='Vx',dtype=float)
    vx = pd.concat([vx, vxmean.loc[time]]).fillna(abs(vx_median)).values
    lag = np.array(np.round((BS_x0.values.flatten()-X_sat.values.flatten())*6371/vx), dtype='timedelta64[s]')
    time = (X_sat.index-lag).round(sampling_time)
    OMNI = pd.DataFrame(columns=omni.columns)
    OMNI = pd.concat([OMNI, omni.loc[time]])
    OMNI.index = X_sat.index
    return OMNI


def find_stagnation_line(Vz, **kwargs):
    if 'Y_mp' and 'Z_mp' in kwargs:
        Y_grid, Z_grid = kwargs['Y_mp'], kwargs['Z_mp']
    else:
        _, Y_grid, Z_grid = make_mp_grid(**kwargs)
    assert Y_grid.shape == Vz.shape, f"The coordinate grids must have the same shape as the flow map, but Y_mp is {Y_grid.shape}, Z_mp is {Z_grid.shape} and the map is {Vz.shape}."

    # --- Step 1: Smooth the data to reduce noise ---
    Vz_smooth = gaussian_filter_nan_datas(Vz, kwargs.get('sigma', 5))  # Utiliser ma fonction

    # --- Step 2: Extract the zero contour (stagnation line) ---
    contours = measure.find_contours(Vz_smooth, level=kwargs.get('level',0))

    if contours:
        # stagnation_line = max(contours, key=len)
        stagnation_line = np.concatenate(contours)
    else:
        stagnation_line = None

    if stagnation_line is not None:
        z_idx, y_idx = stagnation_line[:, 0], stagnation_line[:, 1]
        y_line = [Y_grid[int(i), int(j)] for i, j in zip(z_idx, y_idx)]
        z_line = [Z_grid[int(i), int(j)] for i, j in zip(z_idx, y_idx)]
        dVz_grid = Vz_smooth[:-1,:]-Vz_smooth[1:,:]
        dVz_grid = np.array([np.nan*np.ones(len(Vz[0]))]+list(dVz_grid))
        # If positive, then we go from negative speed at lower z values towards positive speeds towards higher z values,
        # which corresponds to diverging flows, so potentially a Xline
        diverging = np.array([dVz_grid[int(i), int(j)]<0 for i, j in zip(z_idx, y_idx)])
        y_line = list(np.array(y_line)[diverging])
        z_line = list(np.array(z_line)[diverging])
    else:
        y_line, z_line = [], []
    return y_line, z_line


def prepare_data(df, **kwargs):
    df['normalized_logNoverT'] = df.uniform_depthonly_spectro.values
    r, theta, phi = mp_shue1997(df.theta.values, df.phi.values, coord_sys='spherical')
    df['r_MP_Shue'] = r
    df = reposition(df, Rmin=df.r_MP_Shue.values)
    df['dVz'] = df.Vz.values - df.Vz_MSH.values
    if kwargs.get('uniformize_locally', False):
        df = df[df.locally_uniformized_depthonly_spectro.values > kwargs.get('lower_depth', 0.83)]
    else:
        df = df[df.uniform_depthonly_spectro.values > kwargs.get('lower_depth', 0.83)]

    # symetry 1
    if kwargs.get('symetry_cla', True):
        df2 = df.copy()
        df2['Y'] = -df2.Y.values
        df2['normalized_Y'] = -df2.normalized_Y.values
        df2['omni_CLA'] = -df2.omni_CLA.values
        for col in df.columns:
            if 'Vy' in col:
                df2[col] = -df[col].values
        df = pd.concat([df, df2])

    # symetry 2
    if kwargs.get('symetry_coa', True):
        df2 = df.copy()
        df2['Z'] = -df2.Z.values
        df2['normalized_Z'] = -df2.normalized_Z.values
        df2['omni_COA'] = -df2.omni_COA.values
        df2['omni_CLA'] = -df2.omni_CLA.values
        df2['tilt'] = -df2.tilt.values
        df2['Vz'] = -df2.Vz.values
        df2['Vz_MSH'] = -df2.Vz_MSH.values
        df2['dVz'] = -df2.dVz.values
        for col in df.columns:
            if 'Vz' in col:
                df2[col] = -df[col].values
        df = pd.concat([df, df2])

    return df


def knn_weighted_feature_map(
        BL,
        feature_to_map,
        weights_feature,
        Xgrid,  # has the shape of df[['X','Y','Z']].values, but the coordinates of a grid
        k_neighbors=2000,
        eps=1e-6, **kwargs):
    # Sample weight = inverse density
    sample_weights = BL[weights_feature].values
    # Normalize for numerical stability (should be already done, but just in case)
    sample_weights /= np.mean(sample_weights)

    # KNN
    X = BL[['normalized_X', 'normalized_Y', 'normalized_Z']].values.astype(np.float32)
    y = BL[feature_to_map].values.astype(np.float64)
    N = len(BL)
    reg_nn = NearestNeighbors(n_neighbors=k_neighbors)
    reg_nn.fit(X)
    distances, indices = reg_nn.kneighbors(Xgrid)

    # Combine weights
    wd = 1.0 / (distances + eps)
    # Density-based weights (looked up per neighbor)
    wc = sample_weights[indices]
    # Combined weights
    w = wd * wc

    # Weighted Average
    y_neighbors = y[indices]
    mapped_feature = np.sum(w * y_neighbors, axis=1) / np.sum(w, axis=1)

    valid = (np.median(distances, axis=1) <= kwargs.get('median_distance', 3)).astype(float)
    distance_to_barycenter = np.sqrt((np.median(X[indices, 0], axis=1) - Xgrid[:, 0]) ** 2 + (
                np.median(X[indices, 1], axis=1) - Xgrid[:, 1]) ** 2 + (
                                                 np.median(X[indices, 2], axis=1) - Xgrid[:, 2]) ** 2)
    valid *= (distance_to_barycenter <= kwargs.get('max_distance_barycenter', 1)).astype(float)

    return mapped_feature, valid


def get_weights_gaussian(df, list_conditions):
    weights = np.ones(len(df))

    for feature, center, std in list_conditions:
        if feature == 'omni_CLA':
            clas = (df.omni_CLA.values - center + np.pi) % (2 * np.pi) - np.pi + center
            weights *= np.exp(-(clas - center) ** 2 / (2 * std ** 2))
            weights[abs(clas - center) >= std] = 0
        elif feature == 'omni_COA':
            coas = (df.omni_COA.values - center + np.pi / 2) % np.pi - np.pi / 2 + center
            weights *= np.exp(-(coas - center) ** 2 / (2 * std ** 2))
            weights[abs(coas - center) >= std] = 0
        else:
            weights *= np.exp(-(df[feature].values - center) ** 2 / std)
            weights[abs(df[feature].values - center) >= std] = 0

    weights = weights / np.nansum(weights).item()
    df2 = df.copy()
    df2['weights'] = weights
    return df2


