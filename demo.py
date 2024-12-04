from skystar.utils import cal_premutation
from skystar.dataset import selfdata_starsky
from skystar.dataloader import Dataloader
import skystar
from PIL import Image
from skystar.model import Simple_FCN
import skystar.utils as utils
data=selfdata_starsky('data.npz',)
data_test=selfdata_starsky('data.npz',training=False)
dataloader_train = Dataloader(data,1)

dataloader_test = Dataloader(data_test,2,shuffle=False)
model = Simple_FCN()
# model.load_weights('Simple_FCN.npz')
# model.train(dataloader_train,plot=True)
model.check()
