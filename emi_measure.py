from sympy import exp
from astropy import units as u
import numpy as np
from scipy.optimize import fsolve, root
import math as m


def convert_jyperbeam_k(v,bmaj=0.,bmin=0.,fits_img=None):
    """
    calculate the conversion factor from Jy/beam to K
    ref: http://docs.astropy.org/en/stable/api/astropy.units.brightness_temperature.html#astropy.units.brightness_temperature
    this code give the same results as miriad imstat
    :param v: MHz, the freqency of observation
    :param bmaj: deg, the beam major angle
    :param bmin: deg, the beam minor angle
    :param fits_img: optional, the fits image to provide the bmaj and bmin
    :return: the conversion factor from Jy/beam to K
    """

    if fits_img != None:
        fits_open = fits.open(fits_img)
        head = fits_open[0].header
        bmaj = head['BMAJ']
        bmin = head['BMIN']

    bmaj = bmaj*u.deg
    bmin = bmin*u.deg
    fwhm_to_sigma = 1./(8.*np.log(2.))**0.5
    beam_area = 2.*np.pi*(bmaj*bmin*fwhm_to_sigma**2.)
    freq = v*u.MHz
    equiv = u.brightness_temperature(beam_area, freq)
    return u.Jy.to(u.K, equivalencies=equiv)


def cal_tau_tb_write_3freq_no_cal_error_no_template(freq1, freq2, freq3, bmaj1, bmaj2, bmaj3, bmin1, bmin2, bmin3, hii1, hii2, hii3, bkg1, bkg2, bkg3, te, disf, disb, tt1_has, tt2_has, tt3_has):
    """
    calculate the emissivity and optical depths of the target source
    :param freq1: MHz, the central frequency of the first fits image
    :param freq2: MHz, the central frequency of the second fits image
    :param freq3: MHz, the central frequency of the third fits image
    :param bmaj1: degree, the beam major axis length of the first fits image
    :param bmaj2: degree, the beam major axis length of the second fits image
    :param bmaj3: degree, the beam major axis length of the third fits image
    :param bmin1: degree, the beam minor axis length of the first fits image
    :param bmin2: degree, the beam minor axis length of the second fits image
    :param bmin3: degree, the beam minor axis length of the third fits image
    :param hii1: Jy, the flux density of the source region in the first image
    :param hii2: Jy, the flux density of the source region in the second image
    :param hii3: Jy, the flux density of the source region in the third image
    :param bkg1: Jy, the flux density of the background region in the first image
    :param bkg2: Jy, the flux density of the background region in the second image
    :param bkg3: Jy, the flux density of the background region in the third image
    :param te: K, the electron temperature of the target source
    :param disf: kpc, the distance from the target source to the Sun
    :param disb: kpc, the distance from the target source to the Galactic edge
    :param tt1_has: K, the brightness temperature of the source region in the 408 MHz Haslam map scaled to freq1
    :param tt2_has: K, the brightness temperature of the source region in the 408 MHz Haslam map scaled to freq2
    :param tt3_has: K, the brightness temperature of the source region in the 408 MHz Haslam map scaled to freq3
    :return: the optical depth (tau1, tau2, tau3),
             the emissivity from the target source to the Galactic edge (emib1, emib2, emib3),
             the emissivity from the target source to the Sun (emif1, emif2, emif3),
             the missed surface brightness from the target source to the Galactic edge (xb1, xb2, xb3),
             the missed surface brightness from the target source to the Sun (xf1, xf2, xf3).
    """

    tau1_ini = np.arange(0.5, 3.0, 1.0)
    tb1_ini = np.arange(5000., 50000., 20000.)
    tf1_ini = np.arange(5000., 50000., 20000.)
    xb1_ini = np.arange(5000., 50000., 20000.)
    xf1_ini = np.arange(5000., 30000., 10000.)

    print('Start calculating ...')
    roots, z_sum_min = find_roots_3freq_no_template(hii1, hii2, hii3, te, bkg1, bkg2, bkg3, freq1, freq2, freq3, tt1_has, tt2_has, tt3_has, tau1_ini, tb1_ini, tf1_ini, xb1_ini, xf1_ini, si_syn=si_syn)
    print('Calculation finished.')
    # tau1, tau2, tau3, tb1, tb2, tb3, tf1, tf2, tf3, xb1, xb2, xb3, xf1, xf2, xf3
    if roots is np.nan:
        print('I could not find roots. Roots may not exist. I set all roots to be zeros.')
        tau1, tau2, tau3, tb1, tb2, tb3, tf1, tf2, tf3, xb1, xb2, xb3, xf1, xf2, xf3 = np.zeros(15)
    else:
        tau1, tau2, tau3, tb1, tb2, tb3, tf1, tf2, tf3, xb1, xb2, xb3, xf1, xf2, xf3 = roots

    emib1 = tb1 / disb / 1000.
    emib2 = tb2 / disb / 1000.
    emib3 = tb3 / disb / 1000.
    emif1 = tf1 / disf / 1000.
    emif2 = tf2 / disf / 1000.
    emif3 = tf3 / disf / 1000.
    return tau1, tau2, tau3, emib1, emib2, emib3, emif1, emif2, emif3, xb1, xb2, xb3, xf1, xf2, xf3


def find_roots_3freq_no_template(thii1, thii2, thii3, te, tm1, tm2, tm3, f1, f2, f3, tt1, tt2, tt3,
                                 tau1_ini, tb1_ini, tf1_ini, xb1_ini, xf1_ini, si_syn):

    def func(z):
        """
        define the functions of emissivity calculation
        :param z: the optical depths, brightness temperature
        :param template: a vot file containing the best roots from previous calculation, use these roots as the best guess
        if template == None, means no best guess provided, I will use several group of guesses to solve the equations
        :return: the equation set
        """
        tau1, tau2, tau3, tb1, tb2, tb3, tf1, tf2, tf3, xb1, xb2, xb3, xf1, xf2, xf3 = z
        f = np.zeros(15)  # len(f) = 15

        f[0] = te * (1 - exp(-tau1)) + (tb1-xb1) * exp(-tau1) + tf1 - xf1 - thii1
        f[1] = te * (1 - exp(-tau2)) + (tb2-xb2) * exp(-tau2) + tf2 - xf2 - thii2
        f[2] = te * (1 - exp(-tau3)) + (tb3-xb3) * exp(-tau3) + tf3 - xf3 - thii3

        f[3] = tt1 - xf1 - xb1 - tm1
        f[4] = tt2 - xf2 - xb2 - tm2
        f[5] = tt3 - xf3 - xb3 - tm3

        f[6] = tf1 + tb1 - tt1
        f[7] = tf2 + tb2 - tt2
        f[8] = tf3 + tb3 - tt3

        f[8] = tau2 / tau1 - (f2 / f1) ** si_tau
        f[10] = tau3 / tau1 - (f3 / f1) ** si_tau

        f[11] = tb2 / tb1 - (f2 / f1) ** si_syn
        f[12] = tb3 / tb1 - (f3 / f1) ** si_syn

        f[13] = tf2 / tf1 - (f2 / f1) ** si_syn
        f[14] = tf3 / tf1 - (f3 / f1) ** si_syn

        return f

    # define the constants
    b = (f2 / f1) ** si_tau
    c = (f3 / f1) ** si_tau
    d = (f2 / f1) ** si_syn
    e = (f3 / f1) ** si_syn

    # try different initial values of variables
    z_sum_min = 1.e10
    best_roots = np.zeros(15)

    for i in tau1_ini:
        for j in tb1_ini:
            for k in tf1_ini:
                for l in xb1_ini:
                    for m in xf1_ini:
                        init = np.array([i, j, k, l, m])  # tau1, tb1, tf1, x1
                        # initial guess of roots
                        x0 = np.array([init[0], b * init[0], c * init[0],
                                       init[1], d * init[1], e * init[1],
                                       init[2], d * init[2], e * init[2],
                                       init[3], d * init[3], e * init[3],
                                       init[4], d * init[4], e * init[4]])
                        # use fsolve to solve equations
                        # fslove: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fsolve.html
                        roots, infodict, ier, mesg = fsolve(func, x0, xtol=1.49012e-30, full_output=True)
                        z_sum = np.sum(abs(func(roots)))
                        positive = roots > 0.001  # return an array of bool
                        if (z_sum_min > z_sum) and positive.all():
                            z_sum_min = z_sum
                            best_roots = roots

    # check again if roots are real
    if z_sum_min < 1.0:
        return best_roots, z_sum_min
    else:
        print('I can not find roots.')
        return np.nan, z_sum_min


if __name__ == '__main__':

    ###### inputs start ######
    si_syn = -2.7
    gr = 20.  # kpc, the radius of the Galactic disk
    si_tau = -2.1  # the spectral index of the HII region optical depth from Williams & Bridle, 1967, 87, 280
    d_sun = 8.5  # the distance between the Sun and the Galactic center

    freq1 = 76.155  # the central frequency of the first image in MHz
    freq2 = 83.835  # the central frequency of the second image in MHz
    freq3 = 91.515  # the central frequency of the third image in MHz

    # the lengths of the beam major axis in degree for the three images
    bmaj1 = 0.0901480006511
    bmaj2 = 0.0796129599472
    bmaj3 = 0.0724691295069

    # the lengths of the beam minor axis in degree for the three images
    bmin1 = 0.07376939297640001
    bmin2 = 0.06484773213019999
    bmin3 = 0.0589988644441

    #######################################################
    # Parameters below are for HII region G010.769-00.487 #
    #######################################################

    # the average surface brightness in the source region in unit of Jy/beam
    hii1 = 3.530228640345904
    hii2 = 2.678627432563366
    hii3 = 2.1066876082322685
    bkg1 = 6.99128526865497
    bkg2 = 5.134549801612369
    bkg3 = 3.7295525340899363

    te = 5300  # electron temperature of the target source in unit of K
    disf = 5.0  # distance from the target source to the Sun in unit of kpc
    disb = 23.3 # distance from the target source to the Galactic edge along the line of sight in unit of kpc

    tt1_has = 41591.707
    tt2_has = 32088.1
    tt3_has = 25325.809
    ###### inputs end ######

    # find the roots
    roots = cal_tau_tb_write_3freq_no_cal_error_no_template(freq1, freq2, freq3, bmaj1, bmaj2, bmaj3, bmin1, bmin2, bmin3, hii1, hii2, hii3, bkg1, bkg2, bkg3, te, disf, disb, tt1_has, tt2_has, tt3_has)
    tau1, tau2, tau3, emib1, emib2, emib3, emif1, emif2, emif3, xb1, xb2, xb3, xf1, xf2, xf3 = roots

    # compare the roots to confirm that your calculation is correct
    print('The emissivities at three frequencies from the target source to the Galactic edge are:')
    print('Expected emib1 = 1.14, you get', emib1)
    print('Expected emib2 = 0.88, you get', emib2)
    print('Expected emib3 = 0.69, you get', emib3)
    print('The emissivities at three frequencies from the target source to the Sun are:')
    print('Expected emif1 = 3.02, you get', emif1)
    print('Expected emif2 = 2.33, you get', emif2)
    print('Expected emif3 = 1.84, you get', emif3)

