---
layout: post
title: Communicating Data Science to Professionals
---

This summer, I had the pleasure of working at John Hancock Financial as a Data Science Intern. I worked with data scientists, actuaries, clinicians, and fraud analysts to build fraud detection models for long-term care insurance. Though I picked up many technical skills, I learned the most about working with people of different educational backgrounds. Here are 5 of my takeaways about communicating data science to professionals:

1. Less is More
2. Underpromise
3. Talk to People Who Have Touched the Data
4. Ask Precise Questions
5. Create Feedback Loops

<hr>

## Less is More

Always ensure the information you say or show supports the goal or message you
are trying to convey. That goal could include communicating an unexpected
finding, confirming existing beliefs, raising questions, presenting model
results, and more. The supporting visualizations or data summaries you provide
to support your point should be succinct, precise, and worthy of discussion.
Whether it’s a formal presentation or an informal discussion, I think of the
information I convey like an essay: I have one thesis--my point--and
everything I say should support or help progress that point.

Before learning this, I strived to clearly and comprehensively show as much
information as I could. I succeeded in conveying my ideas, but much of the
time, the information I showed did not drive my goal forward and decreased our
overall efficiency. My audience (analysts, clinicians, management) would often
focus on less important parts that I did not intend to discuss at length. When
the discussion goes down such a path, it can be hard to get it back on track,
particularly when senior management is involved.

For example, I once presented a box plot showing the distribution of costs by a
category. My primary goal was to show the trend in the data, with a secondary
goal of showing the distribution. This was not effective for a couple of
reasons. First, a box plot is not a good visual to present to
non-statisticians, as it contains information about quartiles and spread that
are not necessary to convey my point. Second, the team ended up focusing on
explaining the spread of the data as opposed to looking at the actual trend. If
I had focused on showing one idea at a time (i.e. only showing the trend) using
a simpler visual like a point plot, we would have had a more productive
conversation.

Another example is when a fellow data scientist on my team presented her model
results. She briefly showed the distribution of her logistic regression to
explain her scoring system. The team ended up discussing this distribution for
the entire meeting instead of talking about the significant results. Moreover,
management created action items to investigate this distribution, even though
it was not meaningful for the task at hand. Excluding this distribution
entirely would have lead to a more constructive and insightful meeting.

<hr>

## Underpromise

Data science is a relatively new term with a lot of hype surrounding it,
especially to non-practitioners. You must be clear about what data you have and
the power of that data to your clients. No matter how sophisticated your model
is, it will only perform as well as the quality of data you give it. Always be
frank and realistic about what to expect from a model. This way, your team can
have productive and pragmatic discussions about progressing a project. Also,
this allows you to actually deliver on your promises and build trust with the
business. I was surprised to learn how important building trust is as, without
trust, the models you build for the business will never see the light of day.

For example, when I started to build a model to identify suspicious charge
behaviour, I dedicated an entire meeting to deliberate the weaknesses and lack
of granularity in the data. I presented overviews and specific cases of charges
that were reasonable but could not be explained by our digitized data. This was
effective for two reasons. First, the analysts’ and management’s expectations
were lowered to a realistic level, and they clearly understood why and how this
model could fail. Second, my message about data quality resounded with the
clinicians in the room immediately, as they understood the data collection.
This opened up discussions about improving data collection for effective
modelling in the future. After this, we were able to discuss and develop this
model more quickly than past models due to realistic expectations. At the end
of my internship, everyone was pleased with the unexceptional preliminary
results because they understood the limited data.

<hr>

## Talk to People Who Have Touched the Data

It is very easy to try to use the available data at its face value. Even if the
information seems straightforward, I found it valuable to talk to people who
were involved with the data at various stages of its collection. Furthermore,
in a large company, data is often scattered across many places, so there is
likely more relevant data within the company that you are unaware of. You can
only find this data by speaking to individuals outside of your team.

Much of the internal data I had was used by other data scientists on my team,
so we had a general understanding of it. After finding some unusual results, I
decided to seek out and speak to specialists in the company who were involved
in the data collection in some way. I learned a lot of surprising and crucial
information that affected modelling. One particular example was information
about the severities of activities of daily living (ADLs). When someone goes on
claim for long-term care insurance, the company performs an assessment to
determine what kind of care they need in terms of ADLs, which include things
like bathing, dressing, eating, etc. Our data showed 7 (ordered) levels of
severity, but I learned that assessors only have 4 options to choose from. The
7 levels came from wording differences between policies. This was one of many
peculiar but significant pieces of information I learned about the data simply
by reaching out and talking to people within the company.

<hr>

## Ask Precise Questions

Experienced professionals are (obviously) very knowledgeable about their
domain, but their priorities will likely differ from yours as a data scientist.
Information that is important to you may not be important (or may seem obvious)
to them, and vice versa. These varying priorities are useful to gain a new
perspective on the data and problem; however, it means you must be extremely
precise to obtain the information you need.

For example, I was interested in using diagnoses of claimants as a factor to
predict the charges they submitted. When speaking to clinicians, they often
focused on the specificity of the data, the clinical relevance, etc. While this
was important, I was more interested in grouping diseases, understanding the
severities of the diseases, and ways to squeeze more information out of certain
fields. It was not effective to simply approach clinicians with broad questions
about the data. In order to use time efficiently and obtain the most
information, I had to first gain a solid understanding of the data through
exploration and then form precise questions to ask.

Here are some general questions I found helpful when talking about the data:

1. How does the data collection look? Can I see how this data is entered?
2. Can the data be modified by others after it is recorded? If so, under what
   conditions? How can I tell if data has been modified?
3. Which fields depend on other fields when data is inputted? How do the
   various fields interact? Does the meaning of a particular field change based
on other fields?
4. Has the meaning of the data changed in any way (due to rule changes, system
   changes, etc.)? Is there documentation describing this? For example, for an
employee database, perhaps the date an employee started working was entered in
a “Start Date” field. However, after some change, new entries use this “Start
Date” field to record the date on which the employee accepted their offer.
5. Which fields are most susceptible to human error? Why?

<hr>

## Create Feedback Loops

When creating a data solution (or any solution), make sure you set up a
feedback loop. This can be as simple as a comments form or as extensive as
regular meetings. A weekly meeting may not seem useful upon deployment, but I
saw that there are almost always ways to tangibly improve a solution,
particularly within the first few months of deployment.

During my internship, one of the most effective fraud detection systems I
helped develop was a rule-based model in SQL. Essentially, we translated the
decades of experience fraud analysts had into rules in SQL that would score and
flag cases in real time. This creates a queue of prioritized cases for the
fraud analysts to investigate.

The interface the fraud analysts use has a section for feedback, and our
analytics team can see how cases are reviewed and cleared. Every Monday, my
supervisor met with the fraud team to go over cleared cases, discuss potential
tweaks and changes to the implemented rules, and consider new rules. Currently,
the fraud analysts review cases almost exclusively from this queue because of
how effective it has proven. It only reached this state due to the consistent
feedback and improvement from its original form.

<hr>

Overall, I found that effective communication with the business boils down to
having clear goals, opening up many lines of communication, and being precise
with your communication. Though these takeaways came from being a data science
intern working with other professionals, I am sure they apply in some form to
communicating in many technical disciplines.

<hr>

_Note: This post was originally published on medium.com, then moved here on
August 26th, 2020_
