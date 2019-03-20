import autokeras as ak
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.datasets import cifar100
from sklearn.metrics import classification_report

from keras.models import load_model
from keras.utils import plot_model

def main():

    labels = 'beaver,dolphin,otter,seal,whale,aquarium fish,flatfish,ray,shark,trout,orchids,poppies,roses,sunflowers,tulips,bottles,bowls,cans,cups,plates,apples,mushrooms,oranges,pears,sweet peppers,clock,computer keyboard,lamp,telephone,television,bed,chair,couch,table,wardrobe,bee,beetle,butterfly,caterpillar,cockroach,bear,leopard,lion,tiger,wolf,bridge,castle,house,road,skyscraper,cloud,forest,mountain,plain,sea,camel,cattle,chimpanzee,elephant,kangaroo,fox,porcupine,possum,raccoon,skunk,crab,lobster,snail,spider,worm,baby,boy,girl,man,woman,crocodile,dinosaur,lizard,snake,turtle,hamster,mouse,rabbit,shrew,squirrel,maple,oak,palm,pine,willow,bicycle,bus,motorcycle,pickup truck,train,lawn-mower,rocket,streetcar,tank,tractor'
    labelNames = labels.split(',')
    output_path = "report_file"
    
    training_times = [
#         60 * 20
        60 * 60 * 16    # 24 hours
    ]
    
    print('**************data loading **************')
    ((train_x,train_y),(test_x,test_y)) = cifar100.load_data()
    train_x = train_x.astype("float")  / 255.0
    test_x = test_x.astype("float")  / 255.0
    #labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]
    for searching_time in training_times:
        print('*********************** This search will cost {} seconds. ***********************'.format(searching_time))
        
        model = ak.ImageClassifier(path="report_file/{}/".format(searching_time),verbose = True)
        
        model.fit(train_x,train_y,time_limit=searching_time)
        model.final_fit(train_x,train_y,test_x,test_y,retrain=True)
        score = model.evaluate(test_x,test_y)
        predictions = model.predict(test_x)
        report = classification_report(test_y,predictions,target_names=labelNames)
        save = os.path.join(output_path,"{}_seconds_search.txt".format(searching_time))
        f = open(save,"w")
        f.write(report)
        f.write("\nscore:{}".format(score))
        f.close()
        
        model.export_autokeras_model("report_file/{}/{}_model.h5".format(searching_time,searching_time))
        load_model("report_file/{}/{}_model.h5".format(searching_time,searching_time))
        plot_model(model, to_file="report_file/{}/{}_model.png".format(searching_time,searching_time))
        
        
if __name__ == "__main__":
    main()