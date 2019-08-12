import torchvision.transforms as transforms

# im_mean = [0.5, 0.5, 0.5]
# im_std = [0.5, 0.5, 0.5]

im_mean = [0.4914, 0.4822, 0.4465]
im_std = [0.2023, 0.1994, 0.2010]

transforms_c10 = {

    'init': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=im_mean,
                             std=im_std)]),

    'augment': transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(brightness=.3,
                                                       contrast=.1,
                                                       saturation=.2,
                                                       hue=.2)], p=0.5),
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomApply([transforms.RandomRotation(30)], p=0.3),
        transforms.RandomApply([transforms.Resize(35), transforms.RandomCrop(32)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=im_mean,
                             std=im_std)
    ]),

    'augment_simple': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=im_mean,
                             std=im_std)
    ])

}
