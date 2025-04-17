import os
import numpy as np
import pandas as pd
from .data import load_and_scale_returns
from .gan import create_generator, create_discriminator, create_gan

class SyntheticTrainer:
    def __init__(self, input_csv, output_csv,
                 epochs=5000, batch_size=32, hidden=32):
        self.inp = input_csv
        self.out = output_csv
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden = hidden
        os.makedirs(os.path.dirname(self.out), exist_ok=True)

    def run(self):
        real, scaler, length = load_and_scale_returns(self.inp)

        gen  = create_generator(self.hidden)
        disc = create_discriminator(self.hidden)
        gan  = create_gan(gen, disc)

        rng = np.random.default_rng(42)
        for epoch in range(self.epochs+1):
            noise = rng.normal(0,1,(self.batch_size,1))
            synth = gen.predict(noise, verbose=0)

            idx = rng.integers(0, real.shape[0], self.batch_size)
            real_batch = real[idx]

            disc.train_on_batch(real_batch, np.ones((self.batch_size,1)))
            disc.train_on_batch(synth,      np.zeros((self.batch_size,1)))

            noise = rng.normal(0,1,(self.batch_size,1))
            gan.train_on_batch(noise, np.ones((self.batch_size,1)))

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}/{self.epochs}")

        # generate full series
        noise = rng.normal(0,1,(length,1))
        synth_full = gen.predict(noise, verbose=0)
        synth_inv = scaler.inverse_transform(synth_full).flatten()

        pd.DataFrame({"synthetic_return": synth_inv}) \
          .to_csv(self.out, index=False)
        print("Wrote", self.out)
