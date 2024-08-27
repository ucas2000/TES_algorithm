from osgeo import gdal
import numpy as np
import pandas as pd
import pickle


def TES(BT_path, atm_path, outputpath):
    print("TES 函数被调用")
    # 将结果输出为tif影像
    def arr2raster(arr, raster_file, prj=None, trans=None):
        """
        将数组转成栅格文件写入硬盘
        :param arr: 输入的mask数组 ReadAsArray()
        :param raster_file: 输出的栅格文件路径
        :param prj: gdal读取的投影信息 GetProjection()，默认为空
        :param trans: gdal读取的几何信息 GetGeoTransform()，默认为空
        :return:
        """

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(raster_file, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)

        if prj:
            dst_ds.SetProjection(prj)
        if trans:
            dst_ds.SetGeoTransform(trans)

        # 将数组的各通道写入图片
        dst_ds.GetRasterBand(1).WriteArray(arr)

        dst_ds.FlushCache()
        dst_ds = None
        print("successfully convert array to raster")

    # 获取辐亮度对应的亮温
    def obtain_tem(rad, j):
        crf = crf_LUT_list[j]
        tem, rad = min(crf, key=lambda x: abs(rad - float(x[1])))
        return tem

    # 获取亮温对应的辐亮度
    def obtain_rad(tem, j):
        crf = crf_LUT_list[j]
        tem, rad = min(crf, key=lambda x: abs(tem - float(x[0])))
        return rad

    # NEM模块
    def NEM(radsl, atm_down, emis_max):
        global t_ret
        surfrads_nem = np.zeros((n, band_num))
        trad_nem = np.zeros(band_num)
        emiss_ret1 = np.zeros(band_num)
        for j in range(band_num):
            surfrads_nem[0, j] = radsl[j] - (1 - emis_max) * atm_down[i, j]
            trad_nem[j] = surfrads_nem[0, j]
            tTs[j] = obtain_tem(trad_nem[j], j)
        t_nem = np.max(tTs)
        tTs1 = tTs.copy()
        b_t_nem1 = np.zeros(band_num)
        for j in range(band_num):
            tTs1[j] = t_nem
            b_t_nem1[j] = obtain_rad(t_nem, j)
            emiss1[j] = surfrads_nem[0, j] / b_t_nem1[j]
        # 循环n次进行比较
        for x in range(1, n):
            for y in range(band_num):
                surfrads_nem[x, y] = radsl[y] - (1 - emiss1[y]) * atm_down[i, y]
                trad_nem[y] = surfrads_nem[x, y]
            diff_rad = np.zeros(band_num)
            for y in range(band_num):
                diff_rad[y] = abs(surfrads_nem[x, y] - surfrads_nem[x - 1, y])
            if max(diff_rad) < t2:
                for y in range(band_num):
                    tTs1[y] = obtain_tem(trad_nem[y], y)
                t_ret = max(tTs1)
                for y in range(band_num):
                    tTs1[y] = t_ret
                    b_t_nem1[y] = obtain_rad(tTs1[y], y)
                    emiss_ret1[y] = surfrads_nem[x, y] / b_t_nem1[y]
                break
            if x >= 2:
                diff_rad2 = np.zeros(band_num)
                for y in range(band_num):
                    diff_rad2[y] = abs(surfrads_nem[x, y] + surfrads_nem[x - 2, y] - 2 * surfrads_nem[x - 1, y])
                    if min(diff_rad2) > t1:
                        for y in range(band_num):
                            tTs1[y] = obtain_tem(trad_nem[y], y)
                        t_ret = max(tTs1)
                        for y in range(band_num):
                            tTs1[y] = t_ret
                            b_t_nem1[y] = obtain_rad(tTs1[y], y)
                            emiss_ret1[y] = surfrads_nem[x, y] / b_t_nem1[y]
                        break
            # 通过上面两个检验，说明离地辐射中的大气下行辐射得到有效修正，否者就直接用最后一个循环的数据计算温度与发射率
            for y in range(band_num):
                tTs1[y] = obtain_tem(trad_nem[y], y)
            t_ret = max(tTs1)
            for y in range(band_num):
                tTs1[y] = t_ret
                b_t_nem1[y] = obtain_rad(tTs1[y], y)
                emiss_ret1[y] = surfrads_nem[x, y] / b_t_nem1[y]
        t_int = t_ret
        emiss_int = emiss_ret1.copy()
        return t_int, emiss_int

    # RAT模块
    def RAT(emiss_nem_rat):
        beta_rat = np.zeros(band_num)
        avg = emiss_nem_rat.mean()
        for y in range(band_num):
            beta_rat[y] = emiss_nem_rat[y] / avg
        return beta_rat

    # MMD模块
    def func_MMD(radsl, atm_down, beta_mmd):
        emiss_ret_mmd = np.zeros(band_num)
        MMD = max(beta_mmd) - min(beta_mmd)
        # 修改
        emis_min = 0.98743533 - 0.65897189 * MMD ** 0.77074699
        for y in range(band_num):
            emiss_ret_mmd[y] = beta_mmd[y] * emis_min / min(beta_mmd)
        surfrad_mmd = np.zeros((band_num))

        for y in range(band_num):
            surfrad_mmd[y] = (radsl[y] - (1 - emiss_ret_mmd[y]) * atm_down[i, y]) / emiss_ret_mmd[y]
        emis_max_r = max(emiss_ret_mmd)
        emis_max_index = np.argmax(emiss_ret_mmd)
        t_ret = obtain_tem(surfrad_mmd[emis_max_index], emis_max_index)
        return t_ret, emiss_ret_mmd

    # 读取温度查找表
    crf_LUT0 = np.load(r'D:\TES\lut_channel_1.npy')
    crf_LUT1 = np.load(r'D:\TES\lut_channel_2.npy')
    crf_LUT2 = np.load(r'D:\TES\lut_channel_3.npy')

    crf_LUT_list = [crf_LUT0, crf_LUT1, crf_LUT2]

    # 读取多波段tif
    dataset = gdal.Open(BT_path)
    dem_XSize = dataset.RasterXSize  # 列数
    dem_YSize = dataset.RasterYSize  # 行数
    dem_bands = dataset.RasterCount  # 波段数
    transform = dataset.GetGeoTransform()  # 仿射矩阵
    projection = dataset.GetProjection()  # 地图投影信息
    dataset_array = dataset.ReadAsArray()
    z = dataset_array.shape
    #转置
    dat = dataset_array.reshape(z[0], z[1] * z[2]).T
    tgi_all = np.array(dat)

    # 读取大气下行辐射
    # 修改
    # atm_dataset = gdal.Open(atm_path)
    # atm_dataset_array = atm_dataset.ReadAsArray()
    # atm_down = atm_dataset_array.reshape(z[0], z[1] * z[2]).T
    atm_down = np.random.random((z[0], z[1] * z[2])).T


    n = 12  # 最大循环次数
    # 修改
    band_num = 2  # 输入波段数目

    # 构建变量
    tTs = np.zeros(band_num)  # 地表温度
    radsl = np.zeros(band_num)  # 近地表辐亮度
    emiss1 = np.zeros(band_num)  # 发射率
    diff_rad = np.zeros(band_num)  # NEM模块中辐亮度差
    t1 = 0.01
    t2 = 0.01
    #修改
    pixel = 10

    # 储存最终温度结果
    temperature_ret = np.zeros(pixel)
    # 储存发射率结果
    # emissvity_ret = pd.DataFrame(columns=['e1', 'e2', 'e3'])
    # 修改
    emissvity_ret = pd.DataFrame(columns=['e1', 'e2'])

    # 储存QA结果
    qa = np.zeros(pixel)
    zz = 0

    # TES迭代运行
    for i in range(pixel):
        print(format(i / pixel, '.4%'))
        for j in range(band_num):
            # radsl[j] = obtain_rad(tgi_all[i, j], j)
            radsl[j] = (tgi_all[i, j])
        emis_max = 0.99
        t_nem, emiss_nem = NEM(radsl, atm_down, emis_max)
        v1 = 0.00017
        emiss_var = np.var(emiss_nem)
        if emiss_var > v1:  # 非灰体，对应裸土、岩石等
            t_nem, emiss_nem = NEM(radsl, atm_down, 0.96)
            qa[i] = 10
        else:
            emiss_4max = np.array([0.92, 0.95, 0.97, 0.99])
            var4 = np.zeros(len(emiss_4max))
            for ny in range(len(emiss_4max)):
                t_nem, emiss_nem = NEM(radsl, atm_down, emiss_4max[ny])
                var4[ny] = np.var(emiss_nem)
            X = np.vstack([var4 ** 2, var4, np.ones(len(var4))]).T

            # 使用最小二乘法拟合二次多项式
            coeffs = np.linalg.lstsq(X, emiss_4max, rcond=None)[0]

            # 提取拟合系数
            a, b, c = coeffs
            EmissWithMinVar = -b / (2 * a)
            v2 = 0.001
            v3 = 0.001
            v4 = 0.0001
            if 0.9 < EmissWithMinVar < 1:
                d1 = 2 * a * EmissWithMinVar + b
                if abs(d1) < v2:
                    d2 = 2 * a
                    qa[i] = 21
                    if d2 > v3:
                        qa[i] = 31
                        var_min = a * EmissWithMinVar ** 2 + b * EmissWithMinVar + c
                        if var_min < v4:
                            qa[i] = 41
                            t_nem, emiss_nem = NEM(radsl, atm_down, EmissWithMinVar)
                        else:
                            qa[i] = 40
                    else:
                        qa[i] = 30
                else:
                    qa[i] = 20
        beta = RAT(emiss_nem)
        MMD = max(beta) - min(beta)
        t_ret0, emiss_ret0 = func_MMD(radsl, atm_down, beta)
        t_ret_list = np.zeros(12)
        emiss_ret_list = np.zeros([12, band_num])
        t_ret_list[0] = t_ret0
        emiss_ret_list[0] = emiss_ret0
        for nx in range(1, 12):
            t_nem, emiss_nem = NEM(radsl, atm_down, max(locals()[f'emiss_ret{nx - 1}']))
            beta = RAT(emiss_nem)
            locals()[f't_ret{nx}'], locals()[f'emiss_ret{nx}'] = func_MMD(radsl, atm_down, beta)
            t_ret_list[nx] = locals()[f't_ret{nx}']
            emiss_ret_list[nx] = locals()[f'emiss_ret{nx}']
            diff_t = locals()[f't_ret{nx}'] - locals()[f't_ret{nx - 1}']
            if abs(diff_t) < 0.001:
                temperature_ret[i] = locals()[f't_ret{nx}']
                emissvity_ret.loc[i] = locals()[f'emiss_ret{nx}']
                break
            else:
                temperature_ret[i] = locals()[f't_ret{nx}']
                emissvity_ret.loc[i] = locals()[f'emiss_ret{nx}']

        z = 1

    emissvity_ret['ts'] = temperature_ret
    emissvity_ret['qa'] = qa

    # 输出ts_tif影像
    raster_ts = outputpath + r'\ts.tif'
    ts = np.array(emissvity_ret['ts'])
    # ts = ts.reshape((dem_YSize, dem_XSize))
    # arr2raster(ts, raster_ts, prj=projection, trans=transform)
    #
    # # 输出emis1_tif影像
    # raster_emis1 = outputpath + r'\emis1.tif'
    # emis1 = np.array(emissvity_ret['e1'])
    # emis1 = emis1.reshape((dem_YSize, dem_XSize))
    # arr2raster(emis1, raster_emis1, prj=projection, trans=transform)
    #
    # # 输出emis2_tif影像
    # raster_emis2 = outputpath + r'\emis2.tif'
    # emis2 = np.array(emissvity_ret['e2'])
    # emis2 = emis2.reshape((dem_YSize, dem_XSize))
    # arr2raster(emis2, raster_emis2, prj=projection, trans=transform)
    #
    # # 输出emis3_tif影像
    # raster_emis3 = outputpath + r'\emis3.tif'
    # emis3 = np.array(emissvity_ret['e3'])
    # emis3 = emis3.reshape((dem_YSize, dem_XSize))
    # arr2raster(emis3, raster_emis3, prj=projection, trans=transform)
    #
    # # 输出QA_tif影像
    # raster_qa = outputpath + r'\QA.tif'
    # qa = np.array(emissvity_ret['qa'])
    # qa = qa.reshape((dem_YSize, dem_XSize))
    # arr2raster(qa, raster_qa, prj=projection, trans=transform)
    # return 0


# 辐亮度-亮温查找表

def LUT():
    # 提取数据
    CH19_avg = np.loadtxt(r'D:\TES\rtcoef_eos_1_modis-shifted_srf_ch29.txt')
    CH20_avg = np.loadtxt(r'D:\TES\rtcoef_eos_1_modis-shifted_srf_ch31.txt')
    CH21_avg = np.loadtxt(r'D:\TES\rtcoef_eos_1_modis-shifted_srf_ch32.txt')
    data = [
        {"FREQ": CH19_avg[:, 0], "RESP": CH19_avg[:, 1]},
        {"FREQ": CH20_avg[:, 0], "RESP": CH20_avg[:, 1]},
        {"FREQ": CH21_avg[:, 0], "RESP": CH21_avg[:, 1]}
    ]

    lut_three = []
    for channel_data in data:
        lut = []
        resp = channel_data["RESP"]
        freq = 10000/channel_data["FREQ"]
        for j in np.linspace(200, 400, 2001):
            R = planck_radiance_lum(j, freq)
            diff_wl_tmp = np.diff(freq)
            diff_wl_tmp = np.hstack((diff_wl_tmp, [0]))
            lut.append([j, np.sum(resp * R * diff_wl_tmp) / np.sum(resp * diff_wl_tmp)])

        lut_three.append({'look_up': np.array(lut)})
    save_path = r'D:\TES\\'
    for i, lut in enumerate(lut_three):
        np.save(save_path + f'lut_channel_{i + 1}.npy', lut['look_up'])



def planck_T_lum(radiance, lum):
    """
    计算Planck温度。

    参数:
    radiance (float or numpy array): 辐射率
    lum (float): 波长（以纳米为单位）

    返回:
    T (float or numpy array): 计算得到的温度
    """
    # 常数
    c = 2.99792458e+8  # c [m/s]
    h = 6.6260755e-34  # h [J*s]
    k = 1.380658e-23  # k [J/K]
    pi = 3.141592653589793

    # 计算中间常数
    c1_0 = 2 * pi * h * c ** 2
    c1_1 = 2 * h * c ** 2
    c2 = h * c / k

    # 转换波长单位
    lumm = lum / 1000000.0

    # 计算logf和温度T
    logf = c1_1 / (lumm ** 5) / 1000000.0 / radiance + 1.0
    T = c2 / lumm / np.log(logf)

    return T


def planck_radiance_lum(T, lum):
    """
    计算Planck辐射率。

    参数:
    T (float or numpy array): 温度（单位：K）
    lum (float or numpy array): 波长（单位：μm）

    返回:
    radiance (float or numpy array): 辐射率（单位：W m^-2 sr^-1 μm^-1）
    """
    # 常数
    c = 2.99792458e+8  # c [m/s]
    h = 6.6260755e-34  # h [J*s]
    k = 1.380658e-23  # k [J/K]
    pi = 3.141592653589793

    # 计算中间常数
    c1_0 = 2 * pi * h * c ** 2
    c1_1 = 2 * h * c ** 2
    c2 = h * c / k

    # 转换波长单位
    lumm = lum / 1000000.0

    # 计算ca1, ca2, expn和辐射率radiance
    ca1 = c2 / (lumm * T)
    ca2 = c1_1
    expn = np.exp(ca1) - 1.0
    radiance = ca2 / (lumm ** 5) / expn / 1000000.0

    return radiance

# lut_channel_1 = np.load('D:\TES\lut_channel_1.npy')
# lut_channel_2 = np.load('D:\TES\lut_channel_2.npy')
# lut_channel_3 = np.load('D:\TES\lut_channel_3.npy')
# tiff1="D:\TES\DQ1_WSI_N34.68_W96.81_20230815_007095_L20000206363\clip300.tif"
# tiff2="D:\TES\DQ1_WSI_N34.68_W96.81_20230815_007095_L20000206363\clip600.tif"
# outputpath="D:\TES\DQ1_WSI_N34.68_W96.81_20230815_007095_L20000206363"
# TES(tiff1,tiff2,outputpath)