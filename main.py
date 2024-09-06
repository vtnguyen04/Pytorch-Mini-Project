#%%
import torch
import argparse

import config
from dataset import get_data_loaders
from model import get_model
from train import train_fn
from evaluate import evaluate_test, display_misclassified

def main():
    parser = argparse.ArgumentParser(description="Training script với các tham số tùy chỉnh")
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'vit_base_patch16_224'],
                        help='Loại mô hình để training')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE,
                        help='Learning rate cho optimizer')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size cho training')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help='Số epochs để training')
    parser.add_argument('--pretrained', action='store_true',
                        help='Sử dụng pre-trained weights')
    parser.add_argument('--freeze_base', action='store_true',
                        help='Đóng băng base model')
    args = parser.parse_args()

    print(f"Using device: {config.DEVICE}")

    # Cập nhật config với các tham số từ command line
    config.LEARNING_RATE = args.learning_rate
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs

    train_data_loader, val_data_loader, test_data_loader = get_data_loaders()
    model = get_model(args.model, args.pretrained, args.freeze_base)

    print(f"Training {args.model} model")
    print(f"Pretrained: {args.pretrained}, Freeze base: {args.freeze_base}")
    print(f"Learning rate: {config.LEARNING_RATE}, Batch size: {config.BATCH_SIZE}, Epochs: {config.NUM_EPOCHS}")

    train_fn(config.NUM_EPOCHS, train_data_loader, val_data_loader, model)

    model.load_state_dict(torch.load(config.CHECKPOINT_PATH))
    
    evaluate_test(model, test_data_loader)
    display_misclassified(model, test_data_loader)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
# %%

# import torch
# import argparse

# import config
# from dataset import get_data_loaders
# from model import get_model
# from train import train_fn
# from evaluate import evaluate_test, display_misclassified

# def main():
#     parser = argparse.ArgumentParser(description="Training script với các tham số tùy chỉnh")
#     parser.add_argument('--model', type=str, default='resnet18', 
#                         choices=['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'vit_base_patch16_224'],
#                         help='Loại mô hình để training')
#     parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE,
#                         help='Learning rate cho optimizer')
#     parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
#                         help='Batch size cho training')
#     parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
#                         help='Số epochs để training')
#     parser.add_argument('--pretrained', action='store_true',
#                         help='Sử dụng pre-trained weights')
#     parser.add_argument('--freeze_base', action='store_true',
#                         help='Đóng băng base model')
#     args = parser.parse_args()

#     print(f"Using device: {config.DEVICE}")

#     train_data_loader, val_data_loader, test_data_loader = get_data_loaders(batch_size=args.batch_size)
#     model = get_model(args.model, args.pretrained, args.freeze_base)

#     print(f"Training {args.model} model")
#     print(f"Pretrained: {args.pretrained}, Freeze base: {args.freeze_base}")
#     print(f"Learning rate: {args.learning_rate}, Batch size: {args.batch_size}, Epochs: {args.epochs}")

#     train_fn(args.epochs, train_data_loader, val_data_loader, model, learning_rate=args.learning_rate)

#     model.load_state_dict(torch.load(config.CHECKPOINT_PATH))
    
#     evaluate_test(model, test_data_loader)
#     display_misclassified(model, test_data_loader)

# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     main()