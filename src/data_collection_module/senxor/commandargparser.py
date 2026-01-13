# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019 - 2024. All rights reserved.
#
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # camera setup
    # ------------
    parser.add_argument('-com', '--senxor_comport', default=None, type=str,
                        dest='comport', help='Comport connected to thermal camera')
    parser.add_argument('-fps', '--framerate', default=None, type=float,
                        help='Framerate, default 9', dest='fps')
    parser.add_argument('-div', '--framerate_divisor', default=3, type=int,
                        help='Framerate divisor (MI48 register 0xB4), default 3',
                        dest='fps_divisor')
    parser.add_argument('-e', '--emissivity', type=int, default=0.97,
                        dest='emissivity', help='Target emissivity (default 0.97)')
    parser.add_argument('-cis', '--cis_id', default=None, type=int,
                        dest='cis_id', help='Webcam ID for color stream')
    parser.add_argument('-cmap', '--colormap', default='rainbow2', type=str,
                        dest='colormap', help='Thermogram colormap')
    parser.add_argument('-cbar', '--with_colorbar', default=False,
                        action='store_true', help='Display colorbar')
    parser.add_argument('-upscale', '--image_upscale', default=3, type=int,
                        dest='img_scale', help='Upscale image')
    parser.add_argument('-inter', '--interpolation', default=6, dest='interpolation',
                        type=int, help='cv.INTER_*; default 6 (nearest exact)')
    parser.add_argument('-invalid_sn', '--allow_invalid_sn', default=False,
                        action='store_true', help='Allow SN0000000000')

    # ambient conditions
    # ------------------
    parser.add_argument('-Ta', '--ambient_temperature', default=23.0,
                        type=float, dest='Ta',
                        help='Ambient temperature (default 23 deg C)')
    parser.add_argument('-Tbb', '--blackbody_temperature', default=35.0,
                        type=float, dest='Tbb',
                        help='Reference blackbody temperature (default 35.0 deg. C)')
    parser.add_argument('-Rh', '--ambient_humidity', default=70,
                        type=float, dest='Rh', help='Ambient humidity')
    # equipment
    # ---------
    parser.add_argument('-bb', '--blackbody', default=None,
                        dest='blackbody', type=str,
                        help='TCPIP controllable blackbody, e.g. PCN7')

    # ai models:
    parser.add_argument('-aidenoise', '--ai_model_denoise', default=None,
                    dest='ai_model_denoise', type=str,
                    help='File to the denoising model, compatible with cv.dnn backend')

    # debug
    # -----
    parser.add_argument('-V', '--verbose', default='INFO',
                        dest='verbosity', type=str,
                        help='Change the verbosity of the output'),
    args = parser.parse_args()


    return args

