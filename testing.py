from ImageProcessing.TemplateMatching.template_matcher import TemplateMatcher
from baselines import baselines_by_squash as bline 
from spectrogram import Spectrogram
from ProcessingAlgorithms.preprocess.digfile import DigFile
from ImageProcessing.TemplateMatching.Templates.templates import Templates


if __name__ == "__main__":

    df = DigFile("../dig/WHITE_CH4_SHOT/seg00")
    

    spec = Spectrogram(df, 0.0, 60.0e-6, overlap_shift_factor= 1/4, form='db')
    spec.availableData = ['intensity']


    methodsToUse = ['cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    template_matcher = TemplateMatcher(spec, template=Templates.opencv_long_start_pattern5.value, span=200, k=30, methods=methodsToUse)


    # template_matcher.mask_baselines()


    times, velos, scores, methodsUsed = template_matcher.match()


    pcms, axes = template_matcher.spectrogram.plot(min_time=0, min_vel=0, max_vel=10000, cmap='3w_gby')


    pcm = pcms['intensity raw']
    pcm.set_clim(-2, -55)
    template_matcher.add_to_plot(axes, times, velos, scores, methodsUsed, 
                                show_points=True, 
                                show_medoids=False, 
                                verbose=False, 
                                visualize_opacity=False, 
                                show_bounds=False)

