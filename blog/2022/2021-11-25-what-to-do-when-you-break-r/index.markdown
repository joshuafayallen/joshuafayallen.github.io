---
title: What to Do When You Break R
author: Josh Allen
date: '2021-11-25'
slug: what-to-do-when-you-break-r
categories: []
tags: []
subtitle: 'Dealing With C Stack Usage is too close to the limit error'
summary: ''
authors: []
lastmod: '2021-11-25T14:28:28-05:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

<script src="{{< blogdown/postref >}}index_files/twitter-widget/widgets.js"></script>

Hi all, when I first stated using `R` I tried making my website using `blogdown`. While [Alison Hill PhD](https://alison.rbind.io/post/new-year-new-blogdown/) provides an
excellent intro to launching your website. However, I am truly special and managed to mess this process up. For a few days whenever I did anything more computationally intensive than

``` r
rm(list=ls())
library(tidyverse)
```

I would get a nasty error message saying “c stack usage is too close to the limit” and I could not do anything. This would have been fine but at the time I was still taking classes and need to have `R` working to complete the problem sets.

So what did I do to get in the c stack death spiral and how did it end up being fixed? For the former you should not skip steps in [Dr. Hill’s](https://alison.rbind.io/post/new-year-new-blogdown/) post. For the later well to spare you the long arduous process here is what we tried.

-   me consulting stackoverflow & realizing I really f\*\*\*\*k up
-   restarting my computer
-   uninstalling & reinstalling blogdown
-   uninstalling & reinstalling R
-   uninstalling & reinstalling pandocs

After hours of trouble shooting #rstats twitter came to the rescue when this distress signal was sent out.

<blockquote class="twitter-tweet" data-width="550" data-lang="en" data-dnt="true" data-theme="light"><p lang="en" dir="ltr"><a href="https://twitter.com/hashtag/rstats?src=hash&amp;ref_src=twsrc%5Etfw">#rstats</a> world! I have a student who is getting a &quot;c stack usage is too close to the limit&quot; every time when using knitr (even with an R-chunk-free Rmd file) &amp; getting same error when trying to install anything with devtools or remotes. We&#39;ve un/reinstalled R but no luck. Ideas?</p>&mdash; Andrew Heiss 🇺🇦 (@andrewheiss) <a href="https://twitter.com/andrewheiss/status/1358492134039511040?ref_src=twsrc%5Etfw">February 7, 2021</a></blockquote>

So here is what ended up working. To start you will need a super simple Rmd file to test with in a local directory. I suggest starting a new Rmd file with nothing in it other than the YAML header and “test” in the main body or “Lorem Ipsum” if you feel fancy.

In the terminal run the following code

``` {bash
cd ~ 

ls -la
```

Then look for files starting with a period. Okay if you messed up in the initial blogdown setup you are looking for the “.Rprofile” that is causing you the problem. What ended up happening is that you broke all of R by including a recursive function. So restarting and uninstalling R will not kill the function it will be there

<center>

![](https://media.giphy.com/media/l4FB8FfpphPmxdTkA/giphy.gif)

</center>

What you are going to do is open a terminal in Rstudio or otherwise than start running this.

``` {bash
cat.Rprofile
```

than run

``` {bash
cat.zshrc
```

than after that run

    mv. Rprofile .Rprofile-original

Then close out Rstudio and reopen Rstudio. Then try to knit your super simple Rmd file and install a package and doing something fun! Than knit that file.

Hopefully the dreaded C stack usage error is gone. If it is than celebrate
<center>

![](https://media.giphy.com/media/d86kftzaeizO8/giphy.gif)

</center>

because you can use `R` again!!!!
