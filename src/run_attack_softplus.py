import argparse
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, clamp, load_image, make_dir


def get_beta(i, num_iter):
    """
    Helper method for beta growth
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / num_iter)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_iter', type=int, default=1500, help='number of iterations')
    argparser.add_argument('--img', type=str, default='../data/collie4.jpeg', help='image net file to run attack on')
    argparser.add_argument('--target_img', type=str, default='../data/tiger_cat.jpeg',
                           help='imagenet file used to generate target expl')
    argparser.add_argument('--initialize_from_adv', type=str, default=None, help='initialize from the x_adv from ReLu network')
    argparser.add_argument('--lr', type=float, default=0.0003, help='lr')
    argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
    argparser.add_argument('--output_dir', type=str, default='../output/', help='directory to save results to')
    argparser.add_argument('--beta_growth', help='enable beta growth', action='store_true')
    argparser.add_argument('--beta', type=float, default=0.8, help='beta for the soft-plus model')
    argparser.add_argument('--prefactors', nargs=2, default=[1e11, 1e6], type=float,
                           help='prefactors of losses (diff expls, class loss)')
    argparser.add_argument('--method', help='algorithm for expls',
                           choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                    'pattern_attribution', 'grad_times_input'],
                           default='lrp')
    args = argparser.parse_args()

    # options
    device = torch.device("cuda" if args.cuda else "cpu")
    method = getattr(ExplainingMethod, args.method)

    # load model
    data_mean = np.array([0.485, 0.456, 0.406])
    data_std = np.array([0.229, 0.224, 0.225])
    vgg_model = torchvision.models.vgg16(pretrained=True)
    model = ExplainableNet(vgg_model, data_mean=data_mean, data_std=data_std, beta=1000 if args.beta_growth else None)
    if method == ExplainingMethod.pattern_attribution:
        model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
    model = model.eval().to(device)

    # soft-plus model
    softplus_model = ExplainableNet(vgg_model, data_mean=data_mean, data_std=data_std, beta=args.beta)
    if method == ExplainingMethod.pattern_attribution:
        softplus_model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
    softplus_model = softplus_model.eval().to(device)

    # load images
    x = load_image(data_mean, data_std, device, args.img)
    x_target = load_image(data_mean, data_std, device, args.target_img)
    x_adv = x.clone().detach().requires_grad_()

    # produce expls
    org_acc, org_idx = model.classify(x)
    org_expl, _, _ = get_expl(softplus_model, x, method)
    org_expl = org_expl.detach().cpu()

    target_expl, _, _ = get_expl(softplus_model, x_target, method)
    target_expl = target_expl.detach()

    ####
    print("prediction before attack: ", org_idx.item())


    # initialize with the x_adv from the attack to ReLu network
    if args.initialize_from_adv:
        x_adv = torch.load(args.initialize_from_adv).requires_grad_()


    optimizer = torch.optim.Adam([x_adv], lr=args.lr)

    for i in range(args.num_iter):
        if args.beta_growth:
            model.change_beta(get_beta(i, args.num_iter))

        optimizer.zero_grad()

        # calculate loss
        adv_acc, class_idx = model.classify(x_adv)
        adv_expl, _, _ = get_expl(softplus_model, x_adv, method, desired_index=org_idx)
        loss_expl = F.mse_loss(adv_expl, target_expl)
        loss_output = F.mse_loss(adv_acc, org_acc.detach())
        total_loss = args.prefactors[0]*loss_expl + args.prefactors[1]*loss_output

        # update adversarial example
        total_loss.backward()
        optimizer.step()

        # clamp adversarial example
        # Note: x_adv.data returns tensor which shares data with x_adv but requires
        #       no gradient. Since we do not want to differentiate the clamping,
        #       this is what we need
        x_adv.data = clamp(x_adv.data, data_mean, data_std)


        print("Iteration {}: Total Loss: {}, Expl Loss: {}, Output Loss: {}".format(i, total_loss.item(), loss_expl.item(), loss_output.item()))

    # print("Iteration {}: Total Loss: {}, Expl Loss: {}, Output Loss: {}".format(i, total_loss.item(), loss_expl.item(), loss_output.item()))
    print("prediction after attack: ", model(x_adv).argmax().item())

    # test with original model (with relu activations) # temoorarily commented out
    # model.change_beta(None)
    # adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method)

    # save results
    output_dir = make_dir(args.output_dir)
    plot_overview([x_target, x, x_adv], [target_expl, org_expl, adv_expl], data_mean, data_std, filename=f"{output_dir}overview_{org_idx.item()}_{args.method}.png")
    torch.save(x_adv, f"{output_dir}x_{args.method}.pth")


if __name__ == "__main__":
    main()
