import numpy as np
from skimage import io 
from skimage import transform
import sys


k = 4
img_len = 600


def main(argv):
    
    
    
    img = []
    for i in range(415):
        x = io.imread(str(argv[0])+"/"+str(i)+".jpg")
        #x = transform.resize(x,(img_len,img_len,3))
        img.append(x)

    img = np.array(img)
    img = img.reshape((415,-1))
    img_mean = np.mean(img,axis=0)
    
    

    #mean
    '''
    img_sum = np.sum(img,axis = 1)/415
    img_sum = img_sum.reshape((img_len,img_len,3))
    img_sum = img_sum.astype(np.uint8)
    io.imsave("mean.jpg",img_sum)
    exit()
    '''
    U, s, V = np.linalg.svd((img - img_mean).T,full_matrices=False)
    '''
    lull = np.sum(s)
    for i in range(4):
        print(s[i]/lull)
    exit()
    '''
    
    
    #top4
    '''
    for i in range(4):
        z = U[:,i]
        z += np.mean(img)
        z -= np.min(z)
        z /= np.max(z)
        z = (z * 255).astype(np.uint8)
        z = z.reshape((600,600,3))
        #z = z.astype(np.uint8)
        io.imsave("e_face"+str(i)+".jpg",z)
        
        
    exit()
    '''
    #4sample
    '''
    cc = [7,77,41,414]
    for g in cc:
        weights = []

        for i in range(k):
            gg = np.dot((img[g,:] - img_mean).T,U[:,i])
            weights.append(gg)
        
        result = np.zeros((1080000,))
        
        for i in range(len(weights)):
            result += weights[i] * U[:,i]
        
        #result += np.mean(img)
        result += img_mean.T
        result -= np.min(result)
        result /= np.max(result)
        result = (result * 255).astype(np.uint8)
        
        result = result.reshape((img_len,img_len,3))

        io.imsave("reconstruction"+str(g)+".jpg",result)
    '''

    weights = []
    which = int(argv[1][:-4])

    for i in range(k):
        gg = np.dot((img[which,:] - img_mean).T,U[:,i])
        weights.append(gg)
        
    result = np.zeros((1080000,))
        
    for i in range(len(weights)):
        result += weights[i] * U[:,i]
        
    result += img_mean.T
    result -= np.min(result)
    result /= np.max(result)
    result = (result * 255).astype(np.uint8)
        
    result = result.reshape((img_len,img_len,3))

    io.imsave("reconstruction.jpg",result)
    
    

    

    

if __name__ == '__main__':
   main(sys.argv[1:])

