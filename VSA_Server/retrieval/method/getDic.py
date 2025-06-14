from method.utils import calEuclideanDistance
import numpy as np
def caldis(probe_feature,galleryFeature,gallerynumber):
# distance
    Dis = []
    Distemp=[]
    Matrix=[]
    for i in range((gallerynumber)):
        distance = calEuclideanDistance(probe_feature, galleryFeature[i])
        Dis.append(distance)
    # for i in range((gallerynumber)):
    #     temp = []
    #     for j in range(gallerynumber):
    #         distance = calEuclideanDistance(galleryFeature[i], galleryFeature[j])
    #         temp.append(distance)
    #     Matrix.append(temp)
    # np.save('static/Pic/score.npy',Matrix)
    return Dis

