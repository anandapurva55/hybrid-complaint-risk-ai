# Explanation review sheet (blind A/B)

For each case, read the complaint and the decision, then rate Explanation A and B from 1 (poor) to 5 (excellent) on
**faithful** (says nothing the complaint does not support), **useful** (would help a reviewer act) and **clear**, and mark
which you would rather see. The key to which is the LLM is in `explanation_review_key.json`: do not open it until you finish.

## Case 1
**Complaint:** I really liked these cords. i ordered both colors. they are very lush corduroy material and the color is very rich. since i am short, i wear them as ankle length (i am 5'1"). my only issue is that they run very large. i am normally a 25-26 and the 25 was still a bit large. unfortunately, they are all sold out of size 24 :(

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
Rated low dissatisfaction risk (high 4%, medium 15%, low 81%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

**Explanation B**  
The customer enjoyed the lush corduroy material and rich color of the cords but found them to run large, with sizes 24 and 25 being too big for their usual size 25-26.  
The models rated the risk level as low because the customer's feedback was primarily about sizing issues rather than any safety or compliance concerns, and the probabilities strongly supported a low risk level.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 2
**Complaint:** These shipped fast!!! the fit is good. these shorts have some stretch to them, which is nice. there is a green/blue color in the fine detail making them versatile with a few different colors of tops. true to size. the length is perfect for my taste. i am 5'7".

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
Rated low dissatisfaction risk (high 0%, medium 1%, low 99%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

**Explanation B**  
The customer is highly satisfied with the fast shipping, good fit, and versatile color of the shorts.  
The model rated the risk level as low because the probabilities for low risk are very high (99.3%) and the scores from both classical and transformer models are very low.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 3
**Complaint:** This fabric of this top is gorgeous. however, i'm 5' 5" and it hits in a really weird spot. right no the point of the hip that you don't want to accentuate. you could wear it with leggings to a casual thing, but i work in an office so its too short to wear with leggings and too long to wear with jeans. sad, i love the top 90% of it.

**Decision:** high / escalate_immediately (review reasons: low_confidence, models_disagree)

**Explanation A**  
Rated high dissatisfaction risk (high 38%, medium 42%, low 20%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review. A person should look because the models are not confident, and the two models disagree.  
Notes: Review trigger: the models are not confident.; Review trigger: the two models disagree.

**Explanation B**  
The customer loves the fabric of the top but finds it too short for office wear and doesn't fit well with leggings or jeans.  
The model assigned a high risk level due to the low confidence score of 0.199 and disagreement among the models, leading to a final score of 0.588.  
Notes: low_confidence; models_disagree

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 4
**Complaint:** Material is so thin this shirt ripped in three places after wearing only once.

got a size larger than i needed as i have a larger chest and even though it's a bit lose the thread has ripped material on my chest and in the armpits.

**Decision:** high / escalate_immediately (review reasons: models_disagree)

**Explanation A**  
Rated high dissatisfaction risk (high 74%, medium 17%, low 9%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review. A person should look because the two models disagree.  
Notes: Review trigger: the two models disagree.

**Explanation B**  
The customer reports that the shirt is extremely thin and ripped in multiple places after just one wear, despite ordering a larger size.  
The model assigned a high risk level due to the high probability (73.8%) of the complaint indicating a significant quality issue, which aligns with the customer's experience of the shirt ripping easily.  
Notes: models_disagree

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 5
**Complaint:** I love the cut and fit of this t. it's soft and has just enough flow to be comfortable without being unflattering. the gray floral print is very pretty. it's perfect with jeans or shorts in the summer.

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
The customer expresses satisfaction with the cut, fit, softness, and appearance of the t-shirt, mentioning it is perfect for summer wear.  
The model assessed the complaint as low risk due to the positive feedback and absence of any negative or concerning content, supported by the high probability of 0.994 for a low-risk level.  
Notes: 

**Explanation B**  
Rated low dissatisfaction risk (high 0%, medium 0%, low 99%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 6
**Complaint:** First of all, mine arrived in a regular denim shade, not the dark wash shown. also, the style is more like a blazer, and it was huge. i am 5'2" and ordered the petite xs but even xxs would have looked like a baggy blazer. as well, the sleeves came barely past my wrists, they are designed to cuff as the model has done? perhaps on someone taller and younger it would look edgy and chic! the quality is excellent with some nice details. this might be one to try in person, if you can.

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
Rated medium dissatisfaction risk (high 17%, medium 49%, low 34%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

**Explanation B**  
The customer received a denim jacket in a different color and style than advertised, and it was too large despite ordering the correct size. They suggest trying it in person.  
The model assessed a medium risk level due to discrepancies in product description and size, but the quality was praised.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 7
**Complaint:** Love!!!! i was super hesitant to try this dress on, it didn't look great online or on the hanger. but it looked great on! it has a very boho/70s/vintage feel. i am curvy, and it looked fabulous.

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
Rated low dissatisfaction risk (high 1%, medium 2%, low 97%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

**Explanation B**  
The customer was initially hesitant about the dress but found it to be a great fit and loved its boho/70s/vintage style.  
The model assessed the complaint as low risk due to the positive feedback and the absence of any negative or suspicious content.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 8
**Complaint:** The blouse was not flattering on me. it would be for someone with a straight slim body type, it falls mid hips and so it accentuates my hips making them look larger. it's flared so it cannot really be tucked in.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 54%, medium 38%, low 8%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The blouse was not flattering on me. It would be for someone with a straight slim body type, it falls mid hips and so it accentuates my hips making them look larger. It's flared so it cannot really be tucked in.  
The model assigned a high risk level due to the customer's dissatisfaction with the blouse's fit and style, which could lead to potential returns or negative feedback.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 9
**Complaint:** This shirt is thin and shapeless. the quality is h&m esque. unflattering. not a fan

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 77%, medium 20%, low 3%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The customer expressed dissatisfaction with the shirt's thinness, shapelessness, and overall quality, comparing it unfavorably to H&M products.  
The model rated the risk level as high due to the customer's strong negative feedback on the product's quality and fit, with a high final score of 0.87 and a high probability of 0.771.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 10
**Complaint:** This jacket is just too small for me anyway and i am no way big. i am tall and athletic and this thing looked like it had shrunk in the wash. there are so many better fitting other types out there, this went back

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
Rated high dissatisfaction risk (high 66%, medium 30%, low 4%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

**Explanation B**  
The customer is dissatisfied with the jacket's fit and appearance, claiming it is too small and looks shrunken.  
The model assigned a high risk level due to the customer's dissatisfaction with the product's fit and appearance, which is supported by a high final score of 0.807 and a high probability of 0.659.  
Notes: 

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 11
**Complaint:** I wanted to love this shirt and thought i would based on the reviews. it runs really large however and while i love how flows it is i also looked pregnant. i am keeping though- looks nice layered under a jacket or sweater but alone i feel it looked sloppy and big....... which is sometimes just what you want so it is staying in my closet.

**Decision:** high / escalate_immediately (review reasons: low_confidence, models_disagree)

**Explanation A**  
Rated high dissatisfaction risk (high 44%, medium 44%, low 13%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review. A person should look because the models are not confident, and the two models disagree.  
Notes: Review trigger: the models are not confident.; Review trigger: the two models disagree.

**Explanation B**  
The customer was disappointed with the shirt's size, which ran large and made them look pregnant, though they liked the flow and plan to keep it for layering.  
The model assigned a high risk level due to the customer's dissatisfaction with the shirt's fit and appearance, which could indicate potential issues with product quality or sizing consistency.  
Notes: low_confidence; models_disagree

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 12
**Complaint:** This dress has beautiful lines. i loved the design on the material, but the fabric looked very cheap & even was already snagged when it arrived. unfortunately, i must return it.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The customer loved the dress design but found the fabric to be cheap and already snagged upon arrival, leading to a return.  
The model rated the risk level as high due to the customer's dissatisfaction with the fabric quality, which is a significant factor in their decision to return the product.  
Notes: 

**Explanation B**  
Rated high dissatisfaction risk (high 64%, medium 31%, low 5%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 13
**Complaint:** This is a lovely dress - however, there is a weird flap of fabric under the bow that doesn't sit right. it feels a little like a bib. if i could get it to sit right, i would keep it. every angle of the dress is lovely except for head on - which is a small problem.

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
The customer loves the dress but mentions a fabric issue under the bow that feels like a bib, which affects the overall appearance.  
The model rated the risk level as medium because the complaint indicates a specific design flaw that could be a concern for some customers, but it is not a major issue that would affect the majority of customers.  
Notes: 

**Explanation B**  
Rated medium dissatisfaction risk (high 20%, medium 53%, low 28%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 14
**Complaint:** Agree with the other reviewers. this top is excellent. great length, weight & bounce to it, flattering without being clingy. i'm 5'10" with a long torso, so i'm always in the market for tops with longer hemlines and decent material without being too tight or clingy. this one i love! i bought it in black, then went back for the turquoise, then two more... if that doesn't tell you what a great shirt this is, i don't know what will.

**Decision:** low / auto_reply (review reasons: none)

**Explanation A**  
The customer is highly satisfied with the top, praising its length, weight, bounce, and flattering fit. They have purchased multiple colors and recommend the shirt.  
The models rated the risk level as low due to the overwhelmingly positive feedback and high probabilities associated with low risk.  
Notes: 

**Explanation B**  
Rated low dissatisfaction risk (high 0%, medium 1%, low 99%). Suggested action: reply automatically.  
This is a low-tier case because the review looks clearly positive.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 15
**Complaint:** I wanted to like this coat so bad! i loved the bright red color and design, but when i received it, it seemed very cheap and just too large. i returned it.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The customer was disappointed with the quality and size of the coat they received, despite liking its color and design.  
The model rated the risk level as high due to the customer's dissatisfaction with the product quality and size, which are significant factors in the final score.  
Notes: 

**Explanation B**  
Rated high dissatisfaction risk (high 75%, medium 22%, low 3%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 16
**Complaint:** I looked forward to this dress.  it's a lovely design.  it's a heavy material that doesn't have the most flattering fit.  the green lace has snags along the hemlines which is a disappointment.  i'm returning the dress.

**Decision:** medium / manual_review (review reasons: models_disagree)

**Explanation A**  
The customer expresses disappointment with the dress's fit and quality, specifically mentioning snags on the lace, and plans to return it.  
The model rated the risk level as medium because the customer's feedback indicates dissatisfaction with the product's quality and fit, which could potentially lead to more returns or complaints.  
Notes: models_disagree

**Explanation B**  
Rated medium dissatisfaction risk (high 26%, medium 46%, low 27%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral. A person should look because the two models disagree.  
Notes: Review trigger: the two models disagree.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 17
**Complaint:** Love the wine colored cords-- but i tried them and i was disappointed because my normal size was too small. i usually have to size down in pilcro pants but not these. (i'm an 8, and the 30 wasn't going to button!) i love the slight cropped flare! and the color! did i mention the color? but vanity wouldn't allow me to buy them. hopefully i just got a mislabeled pair!

**Decision:** medium / manual_review (review reasons: none)

**Explanation A**  
The customer loves the wine-colored cords and the slight cropped flare but was disappointed that their normal size was too small. They usually have to size down in Pilcro pants but not with these cords. They hope it was just a mislabeled pair.  
The model rated the risk level as medium because the customer expressed dissatisfaction with the sizing, which could indicate a potential issue with product consistency or labeling.  
Notes: 

**Explanation B**  
Rated medium dissatisfaction risk (high 5%, medium 24%, low 71%). Suggested action: send to a person for review.  
This is a medium-tier case because the review looks mixed or neutral.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie

## Case 18
**Complaint:** I really wanted to like this shirt, but it made me look 9 months pregnant. which would be fine if i were pregnant.  it sits wired in the bust and flows out in the front and back. not flattering at all.

**Decision:** high / escalate_immediately (review reasons: none)

**Explanation A**  
The customer expresses dissatisfaction with the shirt's unflattering fit, particularly noting it makes them look 9 months pregnant.  
The model rated the risk level as high due to the high probability (79.4%) of negative feedback related to the product's fit.  
Notes: 

**Explanation B**  
Rated high dissatisfaction risk (high 79%, medium 18%, low 2%). Suggested action: escalate to a senior agent.  
This is a high-tier case because the models put most of the probability on a strongly unhappy review.  
Notes: No review trigger fired.

Ratings (faithful / useful / clear), A: _ / _ / _   B: _ / _ / _   Preferred: A / B / tie
