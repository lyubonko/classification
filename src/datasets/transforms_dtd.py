import torchvision.transforms as transforms

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

fin_scale = 32

transforms_dtd = {

    'init': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Resize(fin_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean,
                             std=imagenet_std)
    ])

}
