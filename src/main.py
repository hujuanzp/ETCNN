import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from thop import profile

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            print(args.save)
            #-------parameter flops-----------
            print(args.model)
            print('Total params: %.2fM' %(sum(p.numel() for p in _model.parameters())/1000000.0))
            # device = torch.device('cpu' if args.cpu else 'cuda')
            # dummy_input = torch.randn(1, 3, 480, 360).to(device)
            # flops, params = profile(_model, (dummy_input, 0))
            # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
            #-------parameter flops-----------
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
