import argparse
import sys
from tokenize import Double
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy

from data import mnist
from model import MyAwesomeModel

sys.path.append('../')
# import helper

train_losses=[]
train_accuracy=[]

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def display(self, image, label):
        print(f'label = {label}')
        plt.imshow(image.squeeze(), cmap='Greys_r')
        # plt.savefig('fig.png')
        plt.show()

    def plotLoss(self):
        """
        Plot the losses.
        """

        plt.plot(train_losses,'-o')
        # plt.plot(eval_losses,'-o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Training'])
        plt.title('Training Loss')

        plt.show()
    
    def train(self):
        print("Training day and night")
        #
        # Add command line args
        #
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.05, type=float)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--save-model', action='store_true', default=False)
        args = parser.parse_args(sys.argv[2:])
        #
        # TODO: Implement training loop here
        #
        model = MyAwesomeModel()
        print(f'{model}')
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
                    # print(f'Validation w/o dropouts')
                    # set model to evaluation mode to turn off drop outs
                    model.eval()
                    ## TODO: Implement the validation pass and print out the validation accuracy
                    images, labels = next(iter(train_set))
                    # Get the class probabilities
                    ps = torch.exp(model(images))
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy = torch.mean(equals.type(torch.FloatTensor))
                    print(f'Epoch {e}, Training accuracy: {accuracy.item()*100}%, training loss = {running_loss/len(train_set)}')
                    #
                    # Save metrics for plotting.
                    #
                    train_losses.append(running_loss/len(train_set))
                    train_accuracy.append(accuracy)
                    # set model back to train mode
                    model.train()

        if args.save_model:
            print(f'Saving model as mnist.pt')
            torch.save(model.state_dict(), "mnist.pt")

        self.plotLoss()
                        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        
        model = MyAwesomeModel()
        print(f'{model}')
        #
        # Load model from storage.
        #
        print(f'Loading model from {args.load_model_from}')
        state_dict = torch.load(args.load_model_from)
        print(state_dict.keys())
        model.load_state_dict(state_dict)

        # Load the test dataset.
        _, test_set = mnist()

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
        print(f'prediction = {ps.argmax()}, label = {labels[0]}')
        if (ps.argmax() == labels[0]):
            print(f'Correct!')
        else:
                print(f'Incorrect!')  

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    
