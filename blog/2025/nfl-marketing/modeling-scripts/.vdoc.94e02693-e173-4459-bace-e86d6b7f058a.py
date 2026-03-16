# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| label: tbl-personnel 
#| tbl-cap: "Common Personnel Packages"
import polars as pl 
import polars.selectors as cs
from great_tables import GT

te_df = pl.DataFrame(
    {'Number of TEs': [1,2,3]}
)

rb_df = pl.DataFrame(
    {'Number of RBs': [1,2,3]}
)
personnel_df = rb_df.join(te_df, how = 'cross')
personnel_df = (rb_df
    .join(te_df, how = 'cross')
    .with_columns(
        pl.concat_str([
            'Number of RBs',
            'Number of TEs'
        ]).alias('Personnel Package'))
    .with_columns(
        pl.col('Personnel Package').str.to_integer().alias('p')
    )   
        .filter(pl.col('p') <= 22)
        .drop('p')
)

GT(personnel_df)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
#| code-fold: true
#| label: mod-import

import arviz as az 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import preliz as pz
import pymc as pm
import seaborn as sns
from pymc_extras.prior import Prior
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from datetime import date
# seed from random.org

seed = 39233615
rng: np.random.Generator = np.random.default_rng(seed = seed)

min_date = date(2018, 4,1)
max_date= date(2021, 9, 1)

#
#
#
#
#
#
#
#

df = (pl.DataFrame({
    'date_week': pl.date_range(min_date, max_date, "1d", eager = True)}
    )
    .filter(pl.col("date_week").dt.weekday() == 1)
    .with_columns(
        pl.col('date_week').dt.year().alias('year'),
        pl.col('date_week').dt.month().alias('month'),
        pl.col('date_week').dt.ordinal_day().alias('day_of_year')
    )
)


#
#
#
#
#
#
#| code-fold: true
n = df.height
x1 = rng.uniform(low = 0.0, high = 1.0, size = n)
x2 = rng.uniform(low = 0.0, high = 1.0, size = n)


df = (
    df
    .with_columns(
        x1_raw = pl.Series(x1), 
        x2_raw = pl.Series(x2)
    )
    .with_columns(
        pl.when(pl.col('x1_raw') > 0.9)
        .then(pl.col('x1_raw'))
        .otherwise((pl.col('x1_raw')/2))
        .alias('x1'),
        pl.when(pl.col('x2_raw') >0.8)
        .then(pl.col("x2_raw"))
        .otherwise(0)
        .alias('x2')
    )
    .drop(['x1_raw', 'x2_raw'])
)

df.columns

long_data = df.unpivot(on = ['x1','x2'], index = 'date_week')

fig, ax = plt.subplots()
sns.lineplot(data = long_data, x = 'date_week', y = 'value', hue = 'variable', alpha = 0.5)

#
#
#
#
#
#
#| code-fold: True
alpha1: float = 0.4 
alpha2: float = 0.2

df = (
    df
    .with_columns(
        pl.col('x1')
        .map_batches(
            lambda s: geometric_adstock(
                x = s.to_numpy(),
                alpha = alpha1,
                l_max = 8,
                normalize=True
            ).eval().flatten(), 
            return_dtype=pl.Float64
        ).alias('x1_adstock'),
        pl.col('x2')
        .map_batches(
            lambda s: geometric_adstock(
                x = s.to_numpy(),
                alpha = alpha2,
                l_max = 8,
                normalize=True
            ).eval().flatten(), 
            return_dtype=pl.Float64
        ).alias('x2_adstock')
    )
)


#
#
#
#
#
#
#| code-fold: true
lam1: float = 4.0
lam2: float = 3.0

df = (
    df
    .with_columns(
        pl.col('x1_adstock')
        .map_batches(
            lambda s: 
            logistic_saturation(
                x = s.to_numpy(),
                lam = lam1
            ).eval(), 
            return_dtype=pl.Float64
        ).alias('x1_saturated_adstock'), 
        pl.col('x2_adstock')
        .map_batches(
            lambda s: 
            logistic_saturation(
                x = s.to_numpy(),
                lam = lam1
            ).eval(), 
            return_dtype=pl.Float64
        ).alias('x2_saturated_adstock')
    )
)


#
#
#
#
#
#
long_effects = df.unpivot(on = cs.starts_with('x'), index = 'date_week')

fig, ax = plt.subplots()
g = sns.FacetGrid(data= long_effects, col = 'variable', col_wrap = 2)
g.map(sns.lineplot, 'date_week', 'value')

#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: true
import math 
df = df.with_columns(
    trend=(
        pl.linear_space(0.0, 50.0, n)  # sequence 0..50 with n samples
        .add(10.0)
        .pow(1.0 / 4.0)
        .sub(1.0)
    ),
    cs=(
        -(2.0 * 2.0 * math.pi * pl.col("day_of_year") / 365.5).sin()
    ),
    cc=(
        (1.0 * 2.0 * math.pi * pl.col("day_of_year") / 365.5).cos()
    ),
).with_columns(
    seasonality=0.5 * (pl.col("cs") + pl.col("cc"))
)

fig, ax = plt.subplots()
sns.lineplot(x="date_week", y="trend", color="C2", label="trend", data=df, ax=ax)
sns.lineplot(
    x="date_week", y="seasonality", color="C3", label="seasonality", data=df, ax=ax
)
ax.legend(loc="upper left")
ax.set(xlabel="date", ylabel=None)
ax.set_title("Trend & Seasonality Components", fontsize=18, fontweight="bold");
#
#
#
#
#
#
#
#
#
#
#| code-fold: true
epsilon = rng.normal(loc = 0.0, scale = 0.25, size = n)
amplitude = 1 
beta_1 = 3.0
beta_2 = 2.0
betas = [beta_1, beta_2]




df = (
    df
    .with_columns(
        pl.lit(2.0).alias('intercept'),
        pl.Series(epsilon).alias('epsilon'), 
        pl.lit(beta_1).alias('beta_1'), 
        pl.lit(beta_2).alias('beta_2'), 
        event_1=(pl.col("date_week") == date(2019, 5,13)).cast(pl.Float64),
        event_2=(pl.col("date_week") == date(2020, 9, 14)).cast(pl.Float64),
    )
    .with_columns(

    )
    .with_columns(
        (
            pl.col('intercept')
            + pl.col('trend')
            + pl.col("seasonality")
            + pl.lit(1.5) * pl.col('event_1')
            + pl.lit(2.5) * pl.col('event_2')
            + pl.col('beta_1') * pl.col('x1_saturated_adstock')
            + pl.col('beta_2') * pl.col('x2_saturated_adstock')
            + pl.col('epsilon')
        ).alias('y')
    )
)

fig, ax = plt.subplots()
sns.lineplot(x="date_week", y="y", color="black", data=df, ax=ax)
ax.set(xlabel="date", ylabel="y (thousands)")


#
#
#
#
#
#
#
#
#
#
#

df_pd = df.to_pandas()
columns_to_keep = [
    "date_week",
    "y",
    "x1",
    "x2",
    "event_1",
    "event_2",
    "day_of_year",
]

mod_df = df_pd[columns_to_keep]

mod_df['t'] = range(n)

#
#
#
#
#
#
#
#
#
#
#
#
#

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()

examp = (
    df
    .with_columns(
        pl.col('x1_saturated_adstock')
        .map_batches(
            lambda s: scaler.fit_transform(s.to_numpy().reshape(-1,1)).ravel(),
            return_dtype = pl.Float64
        ).alias('scaled_version')
    )
)

long = examp.unpivot(on = ['x1_saturated_adstock', 'scaled_version'], index= 'date_week')

fig,ax = plt.subplots()
sns.scatterplot(data = long, x = 'date_week', y= 'value', hue = 'variable')


#
#
#
#
#
#
#

spend_per_channel = mod_df[['x1', 'x2']].sum(axis = 0)

spend_share =  spend_per_channel/spend_per_channel.sum()

sigma_prior = 2 * spend_share.to_numpy()

X = mod_df.drop('y', axis = 1)

y = mod_df['y']
#
#
#
#
#
#
#
my_priors= {
    "intercept": Prior("Normal", mu=0.5, sigma=0.2),
    "adstock_alpha": Prior("Beta", alpha=1, beta=3, dims="channel"),
    "saturation_beta": Prior("HalfNormal", sigma=sigma_prior, dims="channel"),
    "saturation_lam": Prior("Gamma", alpha=3, beta=1, dims="channel"),
    "gamma_control": Prior("Normal", mu=0, sigma=0.05, dims="control"),
    "gamma_fourier": Prior("Laplace", mu=0, b=0.2, dims="fourier_mode"),
    "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=6)),
}

samp_config = {'progressbar': True, 'random_seed': seed, 'nuts_sampler': 'numpyro'}

mmm = MMM(
    model_config=my_priors,
    sampler_config=samp_config,
    date_column="date_week",
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    channel_columns=["x1", "x2"],
    control_columns=["event_1", "event_2", "t"],
    yearly_seasonality=2,
    target_column = 'y'
)
mmm.build_model(X,y)

mmm.add_original_scale_contribution_variable(
    var=[
        "channel_contribution",
        "control_contribution",
        "intercept_contribution",
        "yearly_seasonality_contribution",
        "y",
    ]
)


#
#
#
#
#
#
#

mmm.sample_prior_predictive(X,y)

fig,axes = mmm.plot.prior_predictive()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
fig, axe = plt.subplots()
pz.Beta(alpha = 1, beta = 3).plot_pdf()

#
#
#
#
#
#
#
#| code-fold: true
raw_spend = np.array([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0, 0, 0, 0, 0, 0])

adstock_spend_1 = geometric_adstock(x=raw_spend, alpha=0.20, l_max=8, normalize=True).eval().flatten()
adstock_spend_2 = geometric_adstock(x=raw_spend, alpha=0.50, l_max=8, normalize=True).eval().flatten()
adstock_spend_3 = geometric_adstock(x=raw_spend, alpha=0.80, l_max=8, normalize=True).eval().flatten()

plt.figure(figsize=(10, 6))

plt.plot(raw_spend, marker='o', label='Raw Spend', color='blue')
plt.fill_between(range(len(raw_spend)), 0, raw_spend, color='blue', alpha=0.2)

plt.plot(adstock_spend_1, marker='o', label='Adstock (alpha=0.20)', color='orange')
plt.fill_between(range(len(adstock_spend_1)), 0, adstock_spend_1, color='orange', alpha=0.2)

plt.plot(adstock_spend_2, marker='o', label='Adstock (alpha=0.50)', color='red')
plt.fill_between(range(len(adstock_spend_2)), 0, adstock_spend_2, color='red', alpha=0.2)

plt.plot(adstock_spend_3, marker='o', label='Adstock (alpha=0.80)', color='purple')
plt.fill_between(range(len(adstock_spend_3)), 0, adstock_spend_3, color='purple', alpha=0.2)

plt.xlabel('Weeks')
plt.ylabel('Spend')
plt.title('Geometric Adstock')
plt.legend()
plt.show()

#
#
#
#
#
#
#
fig, axe = plt.subplots()
pz.Beta(1,2 ).plot_pdf(legend = "Infinite Memory: Alpha = 1,Beta = 1")
pz.Beta(1,10).plot_pdf(legend = 'Quick Decay:Alpha = 1, Beta = 10')
pz.Beta(1, 5).plot_pdf(legend = 'Slowish Decay:Alpha = 1,Beta = 5')

#
#
#
#
#
#

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: true
scaled_spend = np.linspace(start=0.0, stop=1.0, num=100)

saturated_spend_1 = logistic_saturation(x=scaled_spend, lam=1).eval()
saturated_spend_2 = logistic_saturation(x=scaled_spend, lam=2).eval()
saturated_spend_4 = logistic_saturation(x=scaled_spend, lam=4).eval()
saturated_spend_8 = logistic_saturation(x=scaled_spend, lam=8).eval()

plt.figure(figsize=(8, 6))
sns.lineplot(x=scaled_spend, y=saturated_spend_1, label="1")
sns.lineplot(x=scaled_spend, y=saturated_spend_2, label="2")
sns.lineplot(x=scaled_spend, y=saturated_spend_4, label="4")
sns.lineplot(x=scaled_spend, y=saturated_spend_8, label="8")

plt.title('Logistic Saturation')
plt.xlabel('Scaled Marketing Spend')
plt.ylabel('Saturated Marketing Spend')
plt.legend(title='Lambda')
plt.show()
#
#
#
#
#
#
#
#
#
#

mmm.fit(X = X, 
        y = y)

#
#
#
#
#
#

mmm.sample_posterior_predictive(X=X, random_seed = seed)

fig, axes = mmm.plot.posterior_predictive(var=["y_original_scale"], hdi_prob=0.94)
sns.lineplot(
    data=df, x="date_week", y="y", color="black", label="Observed", ax=axes[0][0]
);


#
#
#
#
#
#
base_contributions = (
    mmm.idata["posterior"]["intercept_contribution_original_scale"]
    + mmm.idata["posterior"]["control_contribution_original_scale"].sum(dim="control")
    + mmm.idata["posterior"]["yearly_seasonality_contribution_original_scale"]
)

channel_x1 = mmm.idata["posterior"]["channel_contribution_original_scale"].sel(
    channel="x1"
)
channel_x2 = mmm.idata["posterior"]["channel_contribution_original_scale"].sel(
    channel="x2"
)

fig, ax = plt.subplots()

# Stack the contributions
dates = mmm.model.coords["date"]
base_mean = base_contributions.mean(dim=("chain", "draw")).to_numpy()
x1_mean = channel_x1.mean(dim=("chain", "draw")).to_numpy()
x2_mean = channel_x2.mean(dim=("chain", "draw")).to_numpy()

ax.fill_between(dates, 0, base_mean, alpha=0.7, color="gray", label="Base")
ax.fill_between(
    dates,
    base_mean,
    base_mean + x1_mean,
    alpha=0.7,
    color="C0",
    label="Channel x1",
)
ax.fill_between(
    dates,
    base_mean + x1_mean,
    base_mean + x1_mean + x2_mean,
    alpha=0.7,
    color="C1",
    label="Channel x2",
)

# Plot observed
sns.lineplot(data=df, x="date_week", y="y", color="black", label="Observed", ax=ax)

ax.legend(loc="upper left")
ax.set(xlabel="date", ylabel="y")
fig.suptitle("Contribution Breakdown over Time", fontsize=16, fontweight="bold");
#
#
#
