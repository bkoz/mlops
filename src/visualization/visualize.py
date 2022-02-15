import logging
import argparse
import sys
from tokenize import Double
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy
from data import mnist
# from model import MyAwesomeModel
from src.data.data import loadNpz

train_losses=[]
train_accuracy=[]

FORMAT='%(levelname)s %(asctime)s %(message)s'
DATEFMT='%m/%d/%Y %I:%M:%S %p'
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, encoding='utf-8', level=logging.INFO)
        
class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for visualizing a model",
            usage="python visualize.py <model-file> <dataset>"
        )
        # parser.add_argument("command", help="Subcommand to run")
        # args = parser.parse_args(sys.argv[1:2])
        # if not hasattr(self, args.command):
        #     logging.error('Unrecognized command.')
            
        #     parser.print_help()
        #     exit(1)
        # # use dispatch pattern to invoke method with same name
        # getattr(self, args.command)()
        self.modelfile=sys.argv[1:2][0]
        self.dataset = sys.argv[2:3][0]
        print(f'model-file {self.modelfile}, dataset {self.dataset}')
        self.visualize()
    
    def display(self, image, label):
        """
        Display an image and its label.
        """
        logging.info(f'label = {label}')
        plt.imshow(image.squeeze(), cmap='Greys_r')
        # plt.savefig('fig.png')
        plt.show()

    def plotLoss(self):
        """
        Plot the losses.
        """

        plt.plot(train_losses,'-o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Training'])
        plt.title('Training Loss')
        plt.savefig('reports/figures/corruptnmist/training-loss.png')
        plt.show()
    
    def train(self):
        """
        Model training loop.
        """
        logging.debug("Training loop")
        #
        # Add command line args
        #
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.05, type=float)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--save-model', action='store_true', default=False)
        parser.add_argument('--log-level', default="INFO", type=str)

        args = parser.parse_args(sys.argv[2:])
        l = logging.getLogger()
        l.setLevel(args.log_level)
        #
        # TODO: Implement training loop here
        #
        model = MyAwesomeModel()
        logging.debug(f'{model}')
        train_set, _ = mnist()
        image_index = 0
        
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), args.lr)

        for e in range(args.epochs):
            running_loss = 0
            for images, labels in train_set:
                optimizer.zero_grad()
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()        
                running_loss += loss.item()
                
            else:
                # turn off gradients
                with torch.no_grad():
                    model.eval()
                    ## TODO: Implement the validation pass and print out the validation accuracy
                    images, labels = next(iter(train_set))
                    # Get the class probabilities
                    ps = torch.exp(model(images))
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy = torch.mean(equals.type(torch.FloatTensor))
                    logging.info(f'Epoch {e}, Training accuracy: {accuracy.item()*100}%, training loss = {running_loss/len(train_set)}')
                    #
                    # Save metrics for plotting.
                    #
                    train_losses.append(running_loss/len(train_set))
                    train_accuracy.append(accuracy)
                    # set model back to train mode
                    model.train()

        if args.save_model:
            logging.info(f'Saving model as mnist.pt')
            torch.save(model.state_dict(), "models/corruptmnist/mnist.pt")

        self.plotLoss()
                        
    def visualize(self):
        """
        Extract some intermediate representation of the data (your training set) from your cnn. 
        This could be the features just before the final classification layer.
        """
        logging.info("Extracting intermediate representation of the data")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        
        # TODO: Implement evaluation logic here
        
        # model = MyAwesomeModel()
        # logging.debug(f'{model}')
        # #
        # Load model from storage.
        #
        logging.info(f'Loading model from {self.modelfile}')
        model = torch.load(self.modelfile)
        # logging.debug(state_dict.keys())
        # model.load_state_dict(state_dict)

        # Load the test dataset.
        logging.info(f'Loading dataset {self.dataset}')
        test_set = loadNpz(self.dataset)

        model.eval()

        dataiter = iter(test_set)
        images, labels = dataiter.next()
        img = images[0]
        # Convert 2D image to 1D vector
        img = img.view(1, 784)

        # Calculate the class probabilities (softmax) for img
        with torch.no_grad():
            output = model.forward(img)

        ps = torch.exp(output)

        # Plot the image and probabilities
        # helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')
        logging.info(f'prediction = {ps.argmax()}, label = {labels[0]}')
        if (ps.argmax() == labels[0]):
            logging.info(f'Correct!')
        else:
                logging.info(f'Incorrect!')  

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    
