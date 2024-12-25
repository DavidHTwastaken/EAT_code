from modules import audioencoder, prompt
import argparse
import torch

emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']
emo_label_full = ['angry',  'contempt',  'disgusted',
                  'fear',  'happy',  'neutral',  'sad',  'surprised']
latent_dim = 16

def test_emo_mapper(emotype: str):
    mapper = audioencoder.MappingDeepNetwork(latent_dim=16, style_dim=128, num_domains=8, hidden_dim=512)
    y_trg = emo_label.index(emotype)
    y_trg = torch.tensor(y_trg, dtype=torch.long).unsqueeze(0)
    z_trg = torch.randn(latent_dim).unsqueeze(0)
    print(y_trg.shape)
    print(z_trg.shape)
    print(z_trg)
    s_trg = mapper(z_trg, y_trg)
    print(s_trg.shape)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--emotype', type=str, default='neutral')
    args = argparser.parse_args()
    test_emo_mapper(args.emotype)

if __name__ == '__main__':
    main()