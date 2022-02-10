import compressai.models.google as google


class FactorizedPrior(google.FactorizedPrior):
    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x": x,
            "y": y,
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }
