# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019-2024. All rights reserved.
#
import cv2 as cv
import numpy as np

from senxor.utils import remap, HotSpot, get_contour_stats, radialcoord

KELVIN0 = -273.15

FRAME_HEADER_COLUMN_INDEX = {
    # MI48 frame header is composed composed of 16-bit words, unless remarked otherwise
    'FrameID'       : 0,   # frame counter from start (reset)
    'Vdd'           : 1,   # Volts * 1.e4
    'Tdie'          : 2,   # die temperature, centi-Kelvin
    'ms_from_start' : 3,   # two words, ms from start
    'Tmax'          : 5,   # maximum scene temperature, deci-Kelvin
    'Tmin'          : 6,   # minimum scene temperature, deci-Kelvin
}

# recordings from PySenXor (scripts or SenXorTk GUI) include:
# Datetime, 1 row-worth of header, 1 frame-worth of data.
# the *-worth depends on the SenXor FPA shape.
PYSENXOR_COLUMNS = ['Datetime']

# recording from SenXorVew (Qt GUI) have a few more columns with digested data
EVKGUI_COLUMNS = ['Datetime', 'FrameID', 'Vdd', 'Tdie', 'ms_from_start',
                 'cursor_x', 'cursor_y', 'Tcursor']

# the following fields of the frame header are worth parsing during the analysis
PARSED_HEADER_COLUMNS = ['FrameID', 'Vdd', 'Tdie', 'Tmin_hdr', 'Tmax_hdr']

# we must add some meaningful meta data related to the experiment.
# this can be populated manually and/or automatically
# NOTE: the labels below reflect invariant aspects of the experiment,
#       invariant at least over the duration of the given recording.
EXPERIMENT_COLUMNS = ['filestem', 'SN', 'SW', 'FW']


class DataColumns:

    def __init__(self, fpa_nrows, fpa_ncols, software='pysenxor', with_header=True):
        """
        Define the column labels (names) and column indexes in the file.

        This depends on the data logger (which software recorded the data),
        and whether the header was included or not.
        The available software are 'senxortk' or 'pysenxor', or 'senxorview' (QtGUI).
        'senxortk' always records the header.
        """
        self.nrows = fpa_nrows
        self.ncols = fpa_ncols
        self.software = software
        self.with_header = with_header
        self.header_columns = [f'hdr_{i}' for i in range(self.ncols)]
        self.pixel_columns = [f'px_{i}' for i in range(self.ncols * self.nrows)]

    def get_datafile_columns(self):
        """
        File columns include any structured data, header (unparsed), and pixels columns
        """
        columns = []

        if self.software.lower() in ['pysenxor', 'senxortk', 'senxor-mnr']:
            columns += PYSENXOR_COLUMNS

        if self.software.lower() == 'senxorview':
            columns += EVKGUI_COLUMNS

        if self.with_header:
            columns += self.header_columns

        columns += self.pixel_columns

        self.column_labels = columns
        self.column_indexes = range(len(self.column_labels))

        return self.column_labels, self.column_indexes

    def __call__(self):
        labels, indexes = self.get_datafile_columns()
        return labels, indexes


def parse_header_columns(df, columns=FRAME_HEADER_COLUMN_INDEX, inplace=False):
    """
    Take in a data frame `df` with raw frame header columns `hrd_i` and parse these into
    more meaningful values to analyse, e.g. Tdie, Vdd, FrameID, Tmax, Tmin
    Return a frame with the extra `columns` at the front of the frame.
    """
    parsed = {}
    parsed['FrameID'] = df[f'hdr_{columns["FrameID"]}']
    parsed['Vdd'] = df[f'hdr_{columns["Vdd"]}'] * 1.e-4
    parsed['Tdie'] = df[f'hdr_{columns["Tdie"]}'] * 1.e-2 + KELVIN0
    parsed['Tmin_header'] = df[f'hdr_{columns["Tmin"]}'] * 1.e-1 + KELVIN0
    parsed['Tmax_header'] = df[f'hdr_{columns["Tmax"]}'] * 1.e-1 + KELVIN0

    # ms from start is [least significant word, most signifficant word]
    values = df[[f'hdr_{columns["ms_from_start"] + 1}',
                 f'hdr_{columns["ms_from_start"]}']]
    parsed['ms_from_start'] = values.dot(1 << np.array([16, 0]))

    if inplace:
        df = df.assign(**parsed)
        return None
    return df.assign(**parsed)


def reshape_data_to_image_frame(data, nrows, ncols):
    # first, ensure we have a numpy array to work with
    try:
        # assume pd.Series and convert to a numpy array
        data = data.values
    except AttributeError:
        # assume numpy array already
        pass
    # get it to the correct shape of image frames with the correct resolution
    # in each dimension
    if data.shape == (nrows, ncols):
        return data
    if data.shape == (nrows * ncols, ):
        return data.reshape((nrows, ncols))
    if data.shape == (1, nrows * ncols):
        return data.reshape((nrows, ncols))
    if data.ndim == 3 and data.shape[1:3] == (nrows, ncols):
        return data.reshape((data.shape[0], nrows, ncols))
    else:
        msg = f'data shape {data.shape} not in the correct shape '\
                f'(*, {nrows}, {ncols})'
        raise RuntimeError(msg)


class FrameStatistics:

    supported_metrics = [
        'Tmin', 'Tmax', 'Tmean', 'Tmiddle',
        'rowTmin', 'colTmin', 'rowTmax', 'colTmax',
    ]

    def __init__(self, nrows, ncols, mask=None, use_cached=True):
        """
        This class help to extract overall frame statistics (metrics), in a manner that
        is compatible with pandas df.apply(), but handy for pure numpy processinig too.

        Supported are: Tmin, Tmax, Tmean, Tmiddle, rowTmax, colTmax, rowTmin, colTmin.

        If `mask` is not None, then the above statistics pertain to pixels,
        where mask==True.

        `use_cached==True` offers better efficiency

        Usage:

            framestats = FrameStatistics(mi48.nrows, mi48.ncols)
            framestats(data, 'Tmax') # for each of the supported quantities or
            framestats(data, 'Tmin', 'Tmean', 'Tmiddle')
            framestats(data) # this will return all supported metrics.
        """
        self.nrows, self.ncols = nrows, ncols
        if mask is None:
            self.mask = np.ones((nrows, ncols), dtype=bool)
        self.use_cached = use_cached

    def Tmin(self, frame):
        self.min = frame[self.mask].min()
        return self.min

    def Tmax(self, frame):
        self.max = frame[self.mask].max()
        return self.max

    def Tmean(self, frame):
        self.mean = frame[self.mask].mean()
        return self.mean

    def Tmiddle(self, frame):
        if self.use_cached:
            self.middle = 0.5 * (self.max + self.min)
        else:
            self.middle = 0.5 * (frame[self.mask].max() + frame[self.mask].min())
        return self.middle

    def rowTmin(self, frame):
        # note: below we cannot use frame[mask], because the resulting array
        # does not have the same shape as frame.shape and we would not find
        # the actual pixels corresponding to frame[mask].min()
        # therefore, we take the alternative approach is filtering the indexes
        indices = np.where(frame==self.min)
        row, col = [(r,c) for r,c in zip(indices[0], indices[1]) if self.mask[r,c]][0]
        self.row_min, self.col_min = row, col
        return self.row_min

    def colTmin(self, frame):
        # somehow we must impose that this is called _after_ rowTmin
        return self.col_min

    def rowTmax(self, frame):
        # note: below we cannot use frame[mask], because the resulting array
        # does not have the same shape as frame.shape and we would not find
        # the actual pixels corresponding to frame[mask].max()
        # therefore, we take the alternative approach is filtering the indexes
        indices = np.where(frame==self.max)
        row, col = [(r,c) for r,c in zip(indices[0], indices[1]) if self.mask[r,c]][0]
        self.row_max, self.col_max = row, col
        return self.row_max

    def colTmax(self, frame):
        # somehow we must impose that this is called _after_ rowTmax
        return self.col_max

    def __call__(self, frame, metrics=None):
        if metrics is None:
            func = FrameStatistics.supported_metrics

        frame = reshape_data_to_image_frame(frame, self.nrows, self.ncols)

        if isinstance(func, list):
            result = []
            for ff in func:
                rr = getattr(self, ff)(frame)
                result.append(rr)
        else:
            result = getattr(self, func)(frame)
        return result


def segment_frame(frame, threshold,  fore_back, frame_u8=None, mask=None,
                  stats=None, return_all=False, erode_iter=0,
                  reduced_spread_q=(0.0, 0.02, 0.5, 0.98, 1.0)):
    if mask is None:
        mask = np.ones(frame.shape, dtype=bool)
    if frame_u8 is None:
        frame_u8 = remap(frame, new_range=(0, 255),
                         curr_range=(frame[mask].min(), frame[mask].max()))
        frame_u8[mask is False] = 0
    # we're working with computed threshold which selects small variation only
    # in the segment
    # this may fail.
    # we therefore iteration to find a threshold that is low enough (or high
    # enough) to allow some segmentation
    if fore_back == 'HoC':
        th = threshold
        fill = cv.THRESH_BINARY
    else:
        th = 255 - threshold
        fill = cv.THRESH_BINARY_INV
    th_iteration = 0
    contours = None
    while th_iteration < 5:
        #print(f'segmentation threshold {th}, iteration {th_iteration}')
        (return_th, thresh) = cv.threshold(frame_u8, th, 255, fill)
        thresh[mask==0] = 0
        # reduce the contour size
        if erode_iter is not None:
            thresh = cv.erode(thresh, cv.getStructuringElement(cv.MORPH_CROSS,
                                                (3,3)), iterations=erode_iter)
        # find contours
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL,
                                              cv.CHAIN_APPROX_NONE)
        # avoid getting too small contours because of very high/low threshold
        # and a defect pixel which sets the extreme value
        contours = [it for it in contours if len(it) > 9]

        if contours is None:
            th = th - 5 if fore_back=='HoC' else th + 5
            th_iteration += 1
        else:
            break
    # we give up after a few iterations, so contours may still be None
    #print(contours)
    if contours is None:
        print(f'Cannot find contours. Threshold: {th},'\
              f' iteration {th_iteration}')
        return None
    # Process the contours
    all_contours = []
    nContours = len(contours)
    for i_contour, contour in enumerate(sorted(contours, key=cv.contourArea,
                                               reverse=True)):
        _c = get_contour_stats(frame, [contour], minArea=3)
        if len(_c) == 0:
            continue
        hs = HotSpot(0, frame, _c[0][0], _c[0][1], _c[0][2],
                     {'bbox_extension':10})
        hs.osd['area'] = int(np.abs(hs.osd['area']))
        hs.osd['radial_distance'] = radialcoord(*hs.osd["centroid"])
        hs.osd['center_x'] = hs.osd['centroid'][0]
        hs.osd['center_y'] = hs.osd['centroid'][1]
        hs.osd['threshold'] = return_th
        hs.osd['FgBg'] = fore_back
        hs.osd['SegmentID'] = i_contour
        hs.osd['nSegments'] = nContours
        hs.osd['DenseTarget'] = nContours == 1
        # calculate reduced min/max/spread/sdev
        blob = np.sort(frame[hs.out_frames['hs_mask']==255].flatten())
        A = hs.osd['area']
        P = int(2 * np.sqrt(A * np.pi))  # perimeter of the shape, in pixels
        hs.osd['perimeter'] = P
        if A - P >= 9:
            blob = blob[P:]
        hs.osd['reduced_area'] = A - P
        quants = np.quantile(blob, reduced_spread_q)
        hs.osd['reduced_min'] = quants[0]
        hs.osd['reduced_max'] = quants[-1]
        hs.osd['reduced_sdev'] = blob.std()
        hs.osd['reduced_mean'] = blob.mean()
        hs.osd['reduced_median'] = quants[2]  # 0.5 quantile
        if fore_back=='HoC':
            # hot on cold: update bg max for referene
            hs.osd['bg_max'] = frame[hs.out_frames['bg_mask']==255].max()
            hs.osd['reduced_spread'] = quants[-1] - quants[1]
        else:
            # cold on hot: correct the background (average of lowest 12 pixels);
            # update bg max
            hs.osd['bg'] = np.sort(frame[hs.out_frames['bg_mask']==255])[-12:]\
                    .mean()
            hs.osd['bg_max'] = frame[hs.out_frames['bg_mask']==255].max()
            hs.osd['reduced_spread'] = quants[-2] - quants[0]
        all_contours.append([contour, thresh, hs.osd])
    return all_contours


def get_segments_info(frame, thresholds, fore_back_list, mask,
                    reduced_spread_q=None):
    """
    Segment the frame, and return a list of 3-tuples; each tuple corresponds
    to a segment and is made of:
        [0] the contour of the segment (list of points),
        [1] the mask of the segment itself (frame).
        [2] the statistics of the segment (dictionary), and,
    Depending on the scene and how many threshold level are given, there may be
    many segments (tuples) in the list.
    """
    frame_u8 = remap(frame)

    info_all_segments = []

    for th, fb in zip(thresholds, fore_back_list):
        info_segments = []
        info_segments = segment_frame(frame, threshold=th, fore_back=fb,
                                        frame_u8=frame_u8, mask=mask,
                                        return_all=True,
                                        reduced_spread_q=reduced_spread_q)
        info_all_segments += info_segments
    return info_all_segments


class FrameSegmentation:

    def __init__(self, nrows, ncols, thresholds, fore_back_list,
                 mask=None, use_cached=True):
        self.nrows = nrows
        self.ncols = ncols
        self.use_cached = use_cached
        self.thresholds = [int(255 * th) for th in thresholds]
        self.fore_back_list = fore_back_list
        if mask is None:
            self.mask = np.ones((nrows, ncols), dtype=bool)
        self.reduced_spread_q=[0.0, 0.01, 0.5, 0.99, 1.00]
        self.supported_metrics = []


    def __call__(self, frame):
        """
        Segment the frame and return a list of dictionaries, each dictionary
        containing various statistics of the corresponding segment.
        """
        frame = reshape_data_to_image_frame(frame, self.nrows, self.ncols)
        info_all_segments = get_segments_info(frame, self.thresholds,
                                              self.fore_back_list, self.mask,
                                              self.reduced_spread_q)
        if not self.supported_metrics:
            self.supported_metrics = info_all_segments[0][2].keys()

        return [info_segm[2] for info_segm in info_all_segments]

