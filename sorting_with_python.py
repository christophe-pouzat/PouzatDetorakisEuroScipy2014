
import numpy as np
import matplotlib.pyplot as plt
import scipy
plt.ion()

def mad(x):
    """Returns the Median Absolute Deviation of its argument.
    """
    return np.median(np.absolute(x - np.median(x)))*1.4826


def plot_data_list(data_list,
                   time_axes,
                   linewidth=0.2,
                   color='black'):
    """Plots data when individual recording channels make up elements
    of a list.
    
    Parameters
    ----------
    data_list: a list of numpy arrays of dimension 1 that should all
               be of the same length (not checked).
    time_axes: an array with as many elements as the components of
               data_list. The time values of the abscissa.
    linewidth: the width of the lines drawing the curves.
    color: the color of the curves.
    
    Returns
    -------
    Nothing is returned, the function is used for its side effect: a
    plot is generated. 
    """
    nb_chan = len(data_list)
    data_min = [np.min(x) for x in data_list]
    data_max = [np.max(x) for x in data_list]
    display_offset = list(np.cumsum(np.array([0] +
                                             [data_max[i]-
                                              data_min[i-1]
                                             for i in
                                             range(1,nb_chan)])))
    for i in range(nb_chan):
        plt.plot(time_axes,data_list[i]-display_offset[i],
                 linewidth=linewidth,color=color)
    plt.yticks([])
    plt.xlabel("Time (s)")


def peak(x, minimal_dist=15, not_zero=1e-3):
    """Find peaks on one dimensional arrays.
    
    Parameters
    ----------
    x: a one dimensional array on which scipy.signal.fftconvolve can
       be called.
    minimal_dist: the minimal distance between two successive peaks.
    not_zero: the smallest value above which the absolute value of
    the derivative is considered not null.
    
    Returns
    -------
    An array of (peak) indices is returned.
    """
    ## Get the first derivative
    dx = scipy.signal.fftconvolve(x,np.array([1,0,-1])/2.,'same') 
    dx[np.abs(dx) < not_zero] = 0
    dx = np.diff(np.sign(dx))
    pos = np.arange(len(dx))[dx < 0]
    return pos[:-1][np.diff(pos) > minimal_dist]


def cut_sgl_evt(evt_pos,data,before=14, after=30):
    """Cuts an 'event' at 'evt_pos' on 'data'.
        
    Parameters
    ----------
    evt_pos: an integer, the index (location) of the (peak of) the
             event.
    data: a matrix whose rows contains the recording channels.
    before: an integer, how many points should be within the cut
            before the reference index / time given by evt_pos.
    after: an integer, how many points should be within the cut
           after the reference index / time given by evt_pos.
        
    Returns
    -------
    A vector with the cuts on the different recording sites glued
    one after the other. 
    """
    ns = data.shape[0] ## Number of recording sites
    dl = data.shape[1] ## Number of sampling points
    cl = before+after+1 ## The length of the cut
    cs = cl*ns ## The 'size' of a cut
    cut = np.zeros((ns,cl))
    idx = np.arange(-before,after+1)
    keep = idx + evt_pos
    within = np.bitwise_and(0 <= keep, keep < dl)
    kw = keep[within]
    cut[:,within] = data[:,kw].copy()
    return cut.reshape(cs) 
  

def mk_events(positions, data, before=14, after=30):
    """Make events matrix out of data and events positions.
        
    Parameters
    ----------
    positions: a vector containing the indices of the events.
    data: a matrix whose rows contains the recording channels.
    before: an integer, how many points should be within the cut
            before the reference index / time given by evt_pos.
    after: an integer, how many points should be within the cut
           after the reference index / time given by evt_pos.
        
    Returns
    -------
    A matrix with as many rows as events and whose rows are the cuts
    on the different recording sites glued one after the other. 
    """
    res = np.zeros((len(positions),(before+after+1)*data.shape[0]))
    for i,p in enumerate(positions):
        res[i,:] = cut_sgl_evt(p,data,before,after)
    return res 


def plot_events(evts_matrix, 
                n_plot=None,
                n_channels=4,
                events_color='black', 
                events_lw=0.1,
                show_median=True,
                median_color='red',
                median_lw=0.5,
                show_mad=True,
                mad_color='blue',
                mad_lw=0.5):
    """Plot events.
        
    Parameters
    ----------
    evts_matrix: a matrix of events. Rows are events. Cuts from
                 different recording sites are glued one after the
                 other on each row.
    n_plot: an integer, the number of events to plot (if 'None',
            default, all are shown).
    n_channels: an integer, the number of recording channels.
    events_color: the color used to display events. 
    events_lw: the line width used to display events. 
    show_median: should the median event be displayed?
    median_color: color used to display the median event.
    median_lw: line width used to display the median event.
    show_mad: should the MAD be displayed?
    mad_color: color used to display the MAD.
    mad_lw: line width used to display the MAD.
    
    Returns
    -------
    Noting, the function is used for its side effect.
    """
    if n_plot is None:
        n_plot = evts_matrix.shape[0]

    cut_length = evts_matrix.shape[1] // n_channels 
    
    for i in range(n_plot):
        plt.plot(evts_matrix[i,:], color=events_color, lw=events_lw)
    if show_median:
        MEDIAN = np.apply_along_axis(np.median,0,evts_matrix)
        plt.plot(MEDIAN, color=median_color, lw=median_lw)

    if show_mad:
        MAD = np.apply_along_axis(mad,0,evts_matrix)
        plt.plot(MAD, color=mad_color, lw=mad_lw)
    
    left_boundary = np.arange(cut_length,
                              evts_matrix.shape[1],
                              cut_length*2)
    for l in left_boundary:
        plt.axvspan(l,l+cut_length-1,
                    facecolor='grey',alpha=0.5,edgecolor='none')
    plt.xticks([])
    return


def plot_data_list_and_detection(data_list,
                                 time_axes,
                                 evts_pos,
                                 linewidth=0.2,
                                 color='black'):                             
    """Plots data together with detected events.
        
    Parameters
    ----------
    data_list: a list of numpy arrays of dimension 1 that should all
               be of the same length (not checked).
    time_axes: an array with as many elements as the components of
               data_list. The time values of the abscissa.
    evts_pos: a vector containing the indices of the detected
              events.
    linewidth: the width of the lines drawing the curves.
    color: the color of the curves.
    
    Returns
    -------
    Nothing is returned, the function is used for its side effect: a
    plot is generated. 
    """
    nb_chan = len(data_list)
    data_min = [np.min(x) for x in data_list]
    data_max = [np.max(x) for x in data_list]
    display_offset = list(np.cumsum(np.array([0] +
                                             [data_max[i]-
                                              data_min[i-1] for i in
                                             range(1,nb_chan)])))
    for i in range(nb_chan):
        plt.plot(time_axes,data_list[i]-display_offset[i],
                 linewidth=linewidth,color=color)
        plt.plot(time_axes[evts_pos],
                 data_list[i][evts_pos]-display_offset[i],'ro')
    plt.yticks([])
    plt.xlabel("Time (s)")


def mk_noise(positions, data, before=14, after=30, safety_factor=2, size=2000):
    """Constructs a noise sample.
    
    Parameters
    ----------
    positions: a vector containing the indices of the events.
    data: a matrix whose rows contains the recording channels.
    before: an integer, how many points should be within the cut
            before the reference index / time given by evt_pos.
    after: an integer, how many points should be within the cut
           after the reference index / time given by evt_pos.
    safety_factor: a number by which the cut length is multiplied
                   and which sets the minimal distance between the 
                   reference times discussed in the previous
                   paragraph.
    size: the maximal number of noise events one wants to cut (the
          actual number obtained might be smaller depending on the
          data length, the cut length, the safety factor and the
          number of events).
        
    Returns
    -------
    A matrix with as many rows as noise events and whose rows are
    the cuts on the different recording sites glued one after the
    other. 
    """
    sl = before+after+1 ## cut length
    ns = data.shape[0] ## number of recording sites
    i1 = np.diff(positions) ## inter-event intervals
    minimal_length = round(sl*safety_factor)
    ## Get next the number of noise sweeps that can be
    ## cut between each detected event with a safety factor
    nb_i = (i1-minimal_length)//sl
    ## Get the number of noise sweeps that are going to be cut
    nb_possible = min(size,sum(nb_i[nb_i>0]))
    res = np.zeros((nb_possible,sl*data.shape[0]))
    ## Create next a list containing the indices of the inter event
    ## intervals that are long enough
    idx_l = [i for i in range(len(i1)) if nb_i[i] > 0]
    ## Make next an index running over the inter event intervals
    ## from which at least one noise cut can be made
    interval_idx = 0
    ## noise_positions = np.zeros(nb_possible,dtype=numpy.int)
    n_idx = 0
    while n_idx < nb_possible:
        within_idx = 0 ## an index of the noise cut with a long enough
                       ## interval
        i_pos = positions[idx_l[interval_idx]] + minimal_length
        ## Variable defined next contains the number of noise cuts
        ## that can be made from the "currently" considered long-enough
        ## inter event interval
        n_at_interval_idx = nb_i[idx_l[interval_idx]]
        while within_idx < n_at_interval_idx and n_idx < nb_possible:
            res[n_idx,:]= cut_sgl_evt(int(i_pos),data,before,after)
            ## noise_positions[n_idx] = i_pos
            n_idx += 1
            i_pos += sl
            within_idx += 1
        interval_idx += 1
    ## return (res,noise_positions)
    return res


def mk_center_dictionary(positions, data, before=49, after=80):
    """ Computes clusters 'centers' or templates and associated data.
    
    Clusters' centers should be built such that they can be used for 
    subtraction, this implies that we should make them long enough, on
    both side of the peak, to see them go back to baseline. Formal
    parameters before and after bellow should therefore be set to
    larger values than the ones used for clustering. 
    
    Parameters
    ----------
    positions : a vector of spike times, that should all come from the
                same cluster and correspond to reasonably 'clean'
                events.
    data : a data matrix.
    before : the number of sampling point to keep before the peak.
    after : the number of sampling point to keep after the peak.
    
    Returns
    -------
    A dictionary with the following components:
      center: the estimate of the center (obtained from the median).
      centerD: the estimate of the center's derivative (obtained from
               the median of events cut on the derivative of data).
      centerDD: the estimate of the center's second derivative
                (obtained from the median of events cut on the second
                derivative of data).
      centerD_norm2: the squared norm of the center's derivative.
      centerDD_norm2: the squared norm of the center's second
                      derivative.
      centerD_dot_centerDD: the scalar product of the center's first
                            and second derivatives.
      center_idx: an array of indices generated by
                  np.arange(-before,after+1).
     """
    from scipy.signal import fftconvolve
    from numpy import apply_along_axis as apply
    dataD = apply(lambda x:
                  fftconvolve(x,np.array([1,0,-1])/2.,'same'),
                  1, data)
    dataDD = apply(lambda x:
                   fftconvolve(x,np.array([1,0,-1])/2.,'same'),
                   1, dataD)
        
    evts = mk_events(positions, data, before, after)
    evtsD = mk_events(positions, dataD, before, after)
    evtsDD = mk_events(positions, dataDD, before, after)
    evts_median = apply(np.median,0,evts)
    evtsD_median = apply(np.median,0,evtsD)
    evtsDD_median = apply(np.median,0,evtsDD)
    return {"center" : evts_median, 
            "centerD" : evtsD_median, 
            "centerDD" : evtsDD_median, 
            "centerD_norm2" : np.dot(evtsD_median,evtsD_median),
            "centerDD_norm2" : np.dot(evtsDD_median,evtsDD_median),
            "centerD_dot_centerDD" : np.dot(evtsD_median,
                                            evtsDD_median), 
            "center_idx" : np.arange(-before,after+1)}


def mk_aligned_events(positions, data, before=14, after=30):
    """Align events on the central event using first or second order
    Taylor expansion.
    
    Parameters
    ----------
    positions: a vector of indices with the positions of the
               detected events. 
    data: a matrix whose rows contains the recording channels.
    before: an integer, how many points should be within the cut
            before the reference index / time given by positions.
    after: an integer, how many points should be within the cut
           after the reference index / time given by positions.
       
    Returns
    -------
    A tuple whose elements are:
      A matrix with as many rows as events and whose rows are the
      cuts on the different recording sites glued one after the
      other. These events have been jitter corrected using the
      second order Taylor expansion.
      A vector of events positions where "actual" positions have
      been rounded to the nearest index.
      A vector of jitter values.
      
    Details
    ------- 
    (1) The data first and second derivatives are estimated first.
    (2) Events are cut next on each of the three versions of the data.
    (3) The global median event for each of the three versions are
    obtained.
    (4) Each event is then aligned on the median using a first order
    Taylor expansion.
    (5) If this alignment decreases the squared norm of the event
    (6) an improvement is looked for using a second order expansion.
    If this second order expansion still decreases the squared norm
    and if the estimated jitter is larger than 1, the whole procedure
    is repeated after cutting a new the event based on a better peak
    position (7). 
    """
    from scipy.signal import fftconvolve
    from numpy import apply_along_axis as apply
    from scipy.spatial.distance import squareform
    n_evts = len(positions)
    new_positions = positions.copy()
    jitters = np.zeros(n_evts)
    # Details (1)
    dataD = apply(lambda x: fftconvolve(x,np.array([1,0,-1])/2., 'same'),
                  1, data)
    dataDD = apply(lambda x: fftconvolve(x,np.array([1,0,-1])/2.,'same'),
                   1, dataD)
        
    # Details (2)
    evts = mk_events(positions, data, before, after)
    evtsD = mk_events(positions, dataD, before, after)
    evtsDD = mk_events(positions, dataDD, before, after)    
    # Details (3)
    center = apply(np.median,0,evts)
    centerD = apply(np.median,0,evtsD)
    centerD_norm2 = np.dot(centerD,centerD)
    centerDD = apply(np.median,0,evtsDD)
    centerDD_norm2 = np.dot(centerDD,centerDD)
    centerD_dot_centerDD = np.dot(centerD,centerDD)
    # Details (4)
    for evt_idx in range(n_evts):
        # Details (5)
        evt = evts[evt_idx,:]
        evt_pos = positions[evt_idx]
        h = evt - center
        h_order0_norm2 = sum(h**2)
        h_dot_centerD = np.dot(h,centerD)
        jitter0 = h_dot_centerD/centerD_norm2
        h_order1_norm2 = sum((h-jitter0*centerD)**2)
        if h_order0_norm2 > h_order1_norm2:
            # Details (6)
            h_dot_centerDD = np.dot(h,centerDD)
            first = -2*h_dot_centerD + \
              2*jitter0*(centerD_norm2 - h_dot_centerDD) + \
              3*jitter0**2*centerD_dot_centerDD + \
              jitter0**3*centerDD_norm2
            second = 2*(centerD_norm2 - h_dot_centerDD) + \
              6*jitter0*centerD_dot_centerDD + \
              3*jitter0**2*centerDD_norm2
            jitter1 = jitter0 - first/second
            h_order2_norm2 = sum((h-jitter1*centerD- \
                                  jitter1**2/2*centerDD)**2)
            if h_order1_norm2 <= h_order2_norm2:
                jitter1 = jitter0
        else:
            jitter1 = 0
        if abs(round(jitter1)) > 0:
            # Details (7)
            evt_pos -= int(round(jitter1))
            evt = cut_sgl_evt(evt_pos,data=data,
                              before=before, after=after)
            h = evt - center
            h_order0_norm2 = sum(h**2)
            h_dot_centerD = np.dot(h,centerD)
            jitter0 = h_dot_centerD/centerD_norm2
            h_order1_norm2 = sum((h-jitter0*centerD)**2)		      
            if h_order0_norm2 > h_order1_norm2:
                h_dot_centerDD = np.dot(h,centerDD)
                first = -2*h_dot_centerD + \
                  2*jitter0*(centerD_norm2 - h_dot_centerDD) + \
                  3*jitter0**2*centerD_dot_centerDD + \
                  jitter0**3*centerDD_norm2
                second = 2*(centerD_norm2 - h_dot_centerDD) + \
                  6*jitter0*centerD_dot_centerDD + \
                  3*jitter0**2*centerDD_norm2
                jitter1 = jitter0 - first/second
                h_order2_norm2 = sum((h-jitter1*centerD- \
                                      jitter1**2/2*centerDD)**2)
                if h_order1_norm2 <= h_order2_norm2:
                    jitter1 = jitter0
            else:
                jitter1 = 0
        if sum(evt**2) > sum((h-jitter1*centerD-
                              jitter1**2/2*centerDD)**2):
            evts[evt_idx,:] = evt-jitter1*centerD- \
                jitter1**2/2*centerDD
        new_positions[evt_idx] = evt_pos 
        jitters[evt_idx] = jitter1
    return (evts, new_positions,jitters)


def classify_and_align_evt(evt_pos, data, centers,
                           before=14, after=30):
    """Compares a single event to a dictionary of centers and returns
    the name of the closest center if it is close enough or '?', the
    corrected peak position and the remaining jitter.
    
    Parameters
    ----------
    evt_pos : a sampling point at which an event was detected.
    data : a data matrix.
    centers : a centers' dictionary returned by mk_center_dictionary.
    before : the number of sampling point to consider before the peak.
    after : the number of sampling point to consider after the peak.
    
    Returns
    -------
    A list with the following components:
      The name of the closest center if it was close enough or '?'.
      The nearest sampling point to the events peak.
      The jitter: difference between the estimated actual peak
      position and the nearest sampling point.
    """
    cluster_names = np.sort(list(centers))
    n_sites = data.shape[0]
    centersM = np.array([centers[c_name]["center"]\
                         [np.tile((-before <= centers[c_name]\
                                   ["center_idx"]).\
                                   __and__(centers[c_name]["center_idx"] \
                                           <= after), n_sites)]
                                           for c_name in cluster_names])
    evt = cut_sgl_evt(evt_pos,data=data,before=before, after=after)
    delta = -(centersM - evt)
    cluster_idx = np.argmin(np.sum(delta**2,axis=1))    
    good_cluster_name = cluster_names[cluster_idx]
    good_cluster_idx = np.tile((-before <= centers[good_cluster_name]\
                                ["center_idx"]).\
                                __and__(centers[good_cluster_name]\
                                        ["center_idx"] <= after),
                                        n_sites)
    centerD = centers[good_cluster_name]["centerD"][good_cluster_idx]
    centerD_norm2 = np.dot(centerD,centerD)
    centerDD = centers[good_cluster_name]["centerDD"][good_cluster_idx]
    centerDD_norm2 = np.dot(centerDD,centerDD)
    centerD_dot_centerDD = np.dot(centerD,centerDD)
    h = delta[cluster_idx,:]
    h_order0_norm2 = sum(h**2)
    h_dot_centerD = np.dot(h,centerD)
    jitter0 = h_dot_centerD/centerD_norm2
    h_order1_norm2 = sum((h-jitter0*centerD)**2)     
    if h_order0_norm2 > h_order1_norm2:
        h_dot_centerDD = np.dot(h,centerDD)
        first = -2*h_dot_centerD + \
          2*jitter0*(centerD_norm2 - h_dot_centerDD) + \
          3*jitter0**2*centerD_dot_centerDD + \
          jitter0**3*centerDD_norm2
        second = 2*(centerD_norm2 - h_dot_centerDD) + \
          6*jitter0*centerD_dot_centerDD + \
          3*jitter0**2*centerDD_norm2
        jitter1 = jitter0 - first/second
        h_order2_norm2 = sum((h-jitter1*centerD-jitter1**2/2*centerDD)**2)
        if h_order1_norm2 <= h_order2_norm2:
            jitter1 = jitter0
    else:
        jitter1 = 0
    if abs(round(jitter1)) > 0:
        evt_pos -= int(round(jitter1))
        evt = cut_sgl_evt(evt_pos,data=data,
                          before=before, after=after)
        h = evt - centers[good_cluster_name]["center"]\
          [good_cluster_idx]
        h_order0_norm2 = sum(h**2)
        h_dot_centerD = np.dot(h,centerD)
        jitter0 = h_dot_centerD/centerD_norm2
        h_order1_norm2 = sum((h-jitter0*centerD)**2)       
        if h_order0_norm2 > h_order1_norm2:
            h_dot_centerDD = np.dot(h,centerDD)
            first = -2*h_dot_centerD + \
              2*jitter0*(centerD_norm2 - h_dot_centerDD) + \
              3*jitter0**2*centerD_dot_centerDD + \
              jitter0**3*centerDD_norm2
            second = 2*(centerD_norm2 - h_dot_centerDD) + \
              6*jitter0*centerD_dot_centerDD + \
              3*jitter0**2*centerDD_norm2
            jitter1 = jitter0 - first/second
            h_order2_norm2 = sum((h-jitter1*centerD-jitter1**2/2*centerDD)**2)
            if h_order1_norm2 <= h_order2_norm2:
                jitter1 = jitter0
        else:
            jitter1 = 0
    if sum(evt**2) > sum((h-jitter1*centerD-jitter1**2/2*centerDD)**2):
        return [cluster_names[cluster_idx], evt_pos, jitter1]
    else:
        return ['?',evt_pos, jitter1]


def predict_data(class_pos_jitter_list,
                 centers_dictionary,
                 nb_channels=4,
                 data_length=300000):
    """Predicts ideal data given a list of centers' names, positions,
    jitters and a dictionary of centers.
    
    Parameters
    ----------
    class_pos_jitter_list : a list of lists returned by
                            classify_and_align_evt.
    centers_dictionary : a centers' dictionary returned by
                         mk_center_dictionary.
    nb_channels : the number of recording channels.
    data_length : the number of sampling points.
    
    Returns
    -------
    A matrix of ideal (noise free) data with nb_channels rows and
    data_length columns.
    """
    ## Create next a matrix that will contain the results
    res = np.zeros((nb_channels,data_length))
    ## Go through every list element
    for class_pos_jitter in class_pos_jitter_list:
        cluster_name = class_pos_jitter[0]
        if cluster_name != '?':
            center = centers_dictionary[cluster_name]["center"]
            centerD = centers_dictionary[cluster_name]["centerD"]
            centerDD = centers_dictionary[cluster_name]["centerDD"]
            jitter = class_pos_jitter[2]
            pred = center + jitter*centerD + jitter**2/2*centerDD
            pred = pred.reshape((nb_channels,len(center)//nb_channels))
            idx = centers_dictionary[cluster_name]["center_idx"] + \
              class_pos_jitter[1]
            ## Make sure that the event is not too close to the
            ## boundaries
            within = np.bitwise_and(0 <= idx, idx < data_length)
            kw = idx[within]
            res[:,kw] += pred[:,within]
    return res
