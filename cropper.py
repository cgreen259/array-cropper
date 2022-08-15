import numpy as np

def crop(image):    
    """
    Crops a numpy array of surrounding 0 values
    i.e. the reverse of np.pad
    only works for 1D to 5D array
    """
    
    if len(image.shape) == 1:
        xs= np.where(image != 0)
        image = image[min(xs):max(xs)+1]
        
    elif len(image.shape) == 2:
            xs, ys = np.where(image != 0)
            image = image[min(xs):max(xs)+1, min(ys):max(ys)+1]
    
    elif len(image.shape) == 3:
        xs, ys, zs = np.where(image != 0)
        image = image[min(xs):max(xs)+1, min(ys):max(ys)+1, min(zs):max(zs)+1]
    
    elif len(image.shape) == 4:
        xs, ys, zs, zz = np.where(image != 0)
        image = image[min(xs):max(xs)+1, min(ys):max(ys)+1, min(zs):max(zs)+1, min(zz):max(zz)+1]
        
    elif len(image.shape) == 5:
        xs, ys, zs, zz, aa = np.where(image != 0)
        image = image[min(xs):max(xs)+1, min(ys):max(ys)+1, min(zs):max(zs)+1, min(zz):max(zz)+1, min(aa):max(aa)+1]
    
    else:
        print('[Warning!] Array has dimensions > 5, use ncrop')

    return image

def ncrop(image):
    """
    Crops a numpy array of surrounding 0 values
    i.e. the reverse of np.pad
    works on N-dimensional array
    """
    xs = np.where(image != 0)
    
    for i in range(len(xs)):
        image = np.swapaxes(image, 0, i)
        xmin = min(xs[i])
        xmax = max(xs[i])+1
        image = image[xmin:xmax,...]
        image = np.swapaxes(image, 0, i)
    return image


def nancrop(image_):
    """
    Crops a numpy array of surrounding nan values
    i.e. the reverse of np.pad
    only works for 1D to 5D array
    """
    
    coords = np.argwhere(~np.isnan(image_))
    
    if len(image_.shape) == 2:
        xmin = np.min(coords[:,0])
        xmax = np.max(coords[:,0]) + 1
        ymin = np.min(coords[:,1])
        ymax = np.max(coords[:,1]) + 1
        
        image_ = image_[xmin:xmax, ymin:ymax]
    
    elif len(image_.shape) == 3:
        xmin = np.min(coords[:,0])
        xmax = np.max(coords[:,0]) + 1
        ymin = np.min(coords[:,1])
        ymax = np.max(coords[:,1]) + 1
        zmin = np.min(coords[:,2])
        zmax = np.max(coords[:,2]) + 1
        
        image_ = image_[xmin:xmax, ymin:ymax, zmin:zmax]
        
    elif len(image_.shape) == 4:
        xmin = np.min(coords[:,0])
        xmax = np.max(coords[:,0]) + 1
        ymin = np.min(coords[:,1])
        ymax = np.max(coords[:,1]) + 1
        zmin = np.min(coords[:,2])
        zmax = np.max(coords[:,2]) + 1
        amin = np.min(coords[:,3])
        amax = np.max(coords[:,3]) + 1
        
        image_ = image_[xmin:xmax, ymin:ymax, zmin:zmax, amin:amax]
        
    elif len(image_.shape) == 5:
        xmin = np.min(coords[:,0])
        xmax = np.max(coords[:,0]) + 1
        ymin = np.min(coords[:,1])
        ymax = np.max(coords[:,1]) + 1
        zmin = np.min(coords[:,2])
        zmax = np.max(coords[:,2]) + 1
        amin = np.min(coords[:,3])
        amax = np.max(coords[:,3]) + 1
        bmin = np.min(coords[:,4])
        bmax = np.max(coords[:,4]) + 1
        
        image_ = image_[xmin:xmax, ymin:ymax, zmin:zmax, amin:amax, bmin:bmax]
        
    else:
        print('[Warning!] Array has dimensions > 5, cannot compute')
    
    return image_
    

    
    
    