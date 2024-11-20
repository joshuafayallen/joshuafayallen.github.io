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
#| eval: false
import polars as pl
import polars.selectors as cs
from palmerpenguins import load_penguins

penguins = load_penguins().pipe(pl.from_pandas)


pl.Config(tbl_rows = 10)





#
#
#
import polars as pl
import polars.selectors as cs


penguins = pl.read_csv('penguins.csv').with_columns(cs.starts_with('bill').cast(pl.Float64, strict = False)).with_columns(cs.starts_with("flipper").cast(pl.Float64, strict = False)).with_columns(body_mass_g := pl.col('body_mass_g').cast(pl.Float64, strict = False))



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

penguins.head()


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

penguins.glimpse()

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
#
#
#
#| error: true
penguins.filter(pl.col("species") == "Adelie" &
                pl.col("body_mass_g" > mean(pl.col("body_mass_g"))))



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
#| error: true

penguins.filter((pl.col("species") == "Adelie") &
                (pl.col("body_mass_g") > pl.col("body_mass_g").mean()))



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
#| error: true

penguins.filter((pl.col("species").is_in(["Chinstrap", "Gentoo"])) & 
                (pl.col("bill_depth_mm") > pl.col("bill_depth_mm").median()))


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
#
#
#| error: true

penguins.filter((pl.col("species").is_in(["Chinstrap", "Gentoo"]).not_()) & 
                (pl.col("island") != 'Dream'))


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

penguins.select((pl.col("island")))


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
#
#| error: true

penguins.select((pl.col("species", "island")))


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
#| error: true

penguins.select(cs.starts_with("bill"))


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
#
#
#| error: true

penguins = penguins.rename({"bill_length_mm":"BillLengthMm",
                "bill_depth_mm":"BillDepthMm"})


penguins = penguins.rename({"BillLengthMm": "bill_length_mm", 
              "BillDepthMm":"bill_depth_mm"})


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
#
#
#
#
#
#| error: true

penguins.mutate({pl.col("bill_length_mm")^2: "sqr_bill_length"}).select(pl.col("sqr_bill_length"))


#
#
#
#
#
#
#
#
#



penguins.with_columns(sqr_bill_length = pl.col("bill_length_mm")**2).select(pl.col("sqr_bill_length")).head()

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



penguins.with_columns(sqr_bill = pl.col("bill_length_mm")**2).with_columns(og_bill = pl.col("sqr_bill").sqrt()).select(pl.col("sqr_bill", "og_bill", "bill_length_mm")).head(5)


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
#
#
#


penguins.with_columns(female = pl.when(pl.col("sex") == "female").then(True).otherwise(False)).select(["sex", "female"]).head(5)


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
#| error: true

starwars = pl.read_parquet("starwars.parquet")

starwars.with_columns(dog_years = pl.col("birth_year") * 7).with_columns(dog_years_string = pl.col("dog_years").cast(str)).with_columns(comment = pl.col("name") + " is " + pl.col("dog_years_string") + " in dog years ")

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
#| error: true

starwars.glimpse()

starwars.with_columns(dog_years = pl.col("birth_year").str.to_integer(strict = False)*7).with_columns(dog_years_string = pl.col("dog_years").cast(pl.String)).with_columns(comment = pl.col("name") + " is " + pl.col("dog_years_string")  + " in dog years").select(pl.col("name", "dog_years", "comment"))

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

penguins.with_columns(big_peng = pl.when(pl.col("body_mass_g") > pl.col("body_mass_g").mean()).then(True).otherwise(False))
#
#
#
#
#
#
#
#
#

penguins.with_columns(
       body_mass_g_sqr = pl.col('body_mass_g')**2,
                       pl.col('body_mass_g_sqr').sqrt()
)

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
#
#| error: true

penguins.group_by(pl.col("species")).agg(total = pl.count())

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

penguins.group_by(pl.col("species")).agg(count = pl.len(),
                                         mean_flipp = pl.mean("flipper_length_mm"),
                                         median_flipp = pl.median("flipper_length_mm"))

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
#| error: true

penguins.group_by(pl.col("species")).agg(cs.starts_with("bill").mean())


#
#
#
#
#
#
#
#

penguins.group_by(pl.col("species")).agg(cs.starts_with("bill").mean().name.suffix("_mean"),
                                         cs.starts_with("bill").median().name.suffix("_median"))

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
#
#
#
#
#
#
#
#
#

national_dict = {"state": ["Ga", "Ga", "Ga",  "NC", "NC", "NC", "CO", "CO", "CO"], "unemployment":[6,6,8,6,4,3,7,8,9], "year": [2019,2018,2017,2019,2018,2017,2019,2018,2017]}


national_data = pl.DataFrame(national_dict)


library_dict = {"state":["CO", "CO", "CO"], "libraries": [23234,2343234,32342342], "year":[2019,2018,2017]}


library_data = pl.DataFrame(library_dict)



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

joined_data = national_data.join(library_data, on = ["state","year"], how = "left")

joined_data 


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


library_dat = library_data.rename({"state": "state_name"})


national_data.join(library_dat, left_on = ["state", "year"],
               right_on = ["state_name", "year"], how = "left" )


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

a = pl.DataFrame(
       {"a": [1,2],
        "b": [3,4]}
)

b = pl.DataFrame({"a" : [3,4], "b": [5,6]})


pl.concat([a, b], how = "vertical")

#
#
#
#
#
#
#
#
#

c = pl.DataFrame({"chars": ["hello", "lorem"], "chars2":["world","ipsum"]})


pl.concat([a,c], how = "horizontal")

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

relig = pl.read_csv("relig_income.csv")

relig.head()


#
#
#
#
#
#
#
#

relig.melt(id_vars = "religion", variable_name = "income_bracket", value_name = "count")

#
#
#
#
#
#
#
#
#


penguins.pivot(index = "island",columns = "species", values = "body_mass_g",
              aggregate_function="sum")

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
#
#
#
#
#
#

billboards = pl.read_csv("billboard.csv")


billboards.melt(id_vars = "artist",value_vars  = cs.starts_with("wk"),
                variable_name = "week", value_name = "count" )



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

starwars_list = starwars.select(["name", "films", "vehicles", "starships"])

starwars_list.glimpse()


starwars_explode =  starwars_list.explode("films").explode("vehicles").explode("starships")

starwars_explode.head()


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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 


penguins = pd.read_csv("penguins.csv")

sns.set_theme(style = "whitegrid")

#
#
#
#
#
#
#

sns.histplot(data = penguins, x = "body_mass_g")

#
#
#
#
#
#
#
#

sns.histplot(data = penguins, x = "body_mass_g", hue  = "species")

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

sns.histplot(data = penguins, x = "body_mass_g", hue = "species", stat = "density")

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

sns.scatterplot(data = penguins, x = "flipper_length_mm", y = "body_mass_g", hue = "species")

#
#
#
#
#
#
#
#
#

sns.scatterplot(data = penguins, x = "flipper_length_mm", y = "body_mass_g", hue = "species", size = "body_mass_g")

#
#
#
#
#
#
#
#
#

exmp = sns.scatterplot(data = penguins, x = "flipper_length_mm", y = "body_mass_g", hue = "species", size = "body_mass_g")

sns.move_legend(exmp, "upper left", bbox_to_anchor = (1,1))

#
#
#
#
#
#
#
#
#

sns.lmplot(data = penguins, x = "flipper_length_mm", y = "body_mass_g", hue = "species")

#
#
#
#
#
#
#
#
#

labs_examp = sns.lmplot(data = penguins, x = "flipper_length_mm", y = "body_mass_g", hue = "species")



labs_examp.set_axis_labels(x_var= "Flipper Length(mm)", y_var = "Body Mass(g)")


#
#
#
#
#
#
#
#

sns.displot(data = penguins, x = "body_mass_g", hue = "species", row= "species", facet_kws = dict(margin_titles=False))

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

import statsmodels.formula.api as smf
import numpy as np


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

naive = smf.ols("body_mass_g ~ flipper_length_mm", data = penguins).fit()

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
naive.summary()

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


squared = smf.ols("body_mass_g ~ flipper_length_mm**2 + flipper_length_mm  + bill_depth_mm", data = penguins).fit()


squared.summary()

#
#
#
#
#


penguins_pd = pd.read_csv('penguins.csv')


squared = smf.ols('body_mass_g ~ I(flipper_length_mm**2) + flipper_length_mm + bill_depth_mm', data = penguins_pd).fit()




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



interacted = smf.ols("body_mass_g ~ bill_length_mm * species + flipper_length_mm", data = penguins_pd).fit()


interacted.summary()



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
#| eval: false
penugins_pl = pl.read_csv('penguins.csv')

penguins_sans_na = penguins.filter((pl.col("sex").is_not_null())).to_pandas()



with_species = smf.ols('body_mass_g ~ bill_length_mm + flipper_length_mm + species', data = penguins_sans_na).fit()

penguins_sans_na['fitted_vals'] = with_species.fittedvalues

penguins_sans_na['residuals'] = with_species.resid



sns.scatterplot(x = "fitted_vals", y = "residuals", hue = "species", data = penguins_sans_na)




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
#
#
#
#
#
#| error: true

penguins.slice(89:10)


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
#| error: true

penguins.slice(89,10)


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
#
#
#
#
#| error: true

penguins.sample(n = 10, seed = 1994)


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
#| error: true
from janitor import clean_names

penguins = penguins.rename({"bill_length_mm": "BillLengthMm",
                "bill_depth_mm" : "BillDepthMm"})

penguins.clean_names()


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
#| error: true

clean_names(penguins)


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

penguins = pl.read_csv('penguins.csv')

#
#
#
import pandas as pd 

penguins.glimpse()

penguins_pd = penguins.to_pandas()

penguins_clean = clean_names(penguins_pd, case_type = "snake")

penguins = pl.from_pandas(penguins_clean)

penguins.glimpse()


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
#
#

vec1 = penguins["bill_depth_mm"]

print(vec1[0,1])


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
#| error: true

import numpy as np 

print(vec1[0:2])


#
#
#
#
#
