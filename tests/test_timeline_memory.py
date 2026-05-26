from __future__ import annotations

from minicode.timeline_memory import (
    LatestStateMemory,
    SemanticStateIndex,
    StateReasoner,
    build_timeline_context,
    date_key,
    extract_question_event_phrases,
    extract_semantic_state_records,
    extract_state_records,
    parse_date,
    score_event_phrase,
    tokenize,
)


def test_tokenize_removes_common_question_words():
    assert tokenize("What was my latest project budget?") == ["latest", "project", "budget"]


def test_date_key_sorts_iso_dates():
    assert date_key("2024-01-02") < date_key("2024-03-01")


def test_parse_date_accepts_longmemeval_format():
    parsed = parse_date("2024/03/08 (Fri) 12:30")
    assert parsed is not None
    assert parsed.year == 2024
    assert parsed.month == 3
    assert parsed.day == 8


def test_build_timeline_context_puts_relevant_latest_candidate_first():
    sessions = [
        [{"role": "user", "content": "My 5K personal best is 26:30."}],
        [{"role": "user", "content": "Update: my 5K personal best is now 25:50."}],
    ]
    context = build_timeline_context(
        question="What was my personal best time in the 5K?",
        sessions=sessions,
        session_ids=["old", "new"],
        session_dates=["2024-01-01", "2024-02-01"],
        ranked_session_ids=["old", "new"],
        top_k_sessions=2,
    )

    assert context.latest_candidates[0].session_id == "new"
    assert "25:50" in context.text
    assert "Latest State Memory" in context.text
    chronological = context.text.split("Chronological evidence:", 1)[1]
    assert chronological.index("2024-01-01") < chronological.index("2024-02-01")


def test_build_timeline_context_respects_ranked_session_filter():
    sessions = [
        [{"role": "user", "content": "The relevant budget is $400."}],
        [{"role": "user", "content": "The relevant budget is $900."}],
    ]
    context = build_timeline_context(
        question="What is the relevant budget?",
        sessions=sessions,
        session_ids=["selected", "filtered"],
        session_dates=["2024-01-01", "2024-02-01"],
        ranked_session_ids=["selected"],
        top_k_sessions=1,
    )

    assert "$400" in context.text
    assert "$900" not in context.text


def test_extract_state_records_from_update_sentence():
    records = extract_state_records(
        "Update: my personal best time is now 25:50.",
        date="2024-02-01",
        evidence_id="session-2:0",
    )

    assert records
    assert records[0].subject == "user"
    assert records[0].attribute == "personal best time"
    assert records[0].value == "25:50"
    assert records[0].confidence > 0.8


def test_latest_state_memory_keeps_newest_value():
    old = extract_state_records(
        "My personal best time is 26:30.",
        date="2024-01-01",
        evidence_id="old",
    )
    new = extract_state_records(
        "My personal best time is now 25:50.",
        date="2024-02-01",
        evidence_id="new",
    )

    latest = LatestStateMemory(old + new).latest_by_key()
    record = latest[("user", "personal best time")]
    assert record.value == "25:50"
    assert "new" in LatestStateMemory(old + new).format_for_prompt()


def test_semantic_state_extractor_captures_value_statement():
    records = extract_semantic_state_records(
        "I recently set a personal best time in a charity 5K run with a time of 25:50.",
        date="2024-02-01",
        evidence_id="run:4",
    )

    assert any("charity 5k run" in record.attribute and record.value == "25:50" for record in records)


def test_semantic_state_index_ranks_question_relevant_events():
    records = extract_semantic_state_records(
        "I started playing along to my favorite songs on my old keyboard today.",
        date="2024-03-01",
        evidence_id="music:0",
    ) + extract_semantic_state_records(
        "I attended a friends and family sale at Nordstrom yesterday.",
        date="2024-03-02",
        evidence_id="shop:0",
    )

    ranked = SemanticStateIndex(records).search(
        "When did I start playing along to my favorite songs on my old keyboard?",
        max_records=2,
    )
    assert ranked
    assert ranked[0].evidence_id == "music:0"
    assert ranked[0].record_type == "event"


def test_state_reasoner_answers_date_difference():
    records = [
        *extract_semantic_state_records(
            "I visited the Museum of Modern Art today.",
            date="2024/03/01 (Fri)",
            evidence_id="moma:0",
        ),
        *extract_semantic_state_records(
            "I attended the Ancient Civilizations exhibit at the Metropolitan Museum of Art.",
            date="2024/03/08 (Fri)",
            evidence_id="met:0",
        ),
    ]

    result = StateReasoner(records).answer(
        "How many days passed between my visit to the Museum of Modern Art and the Ancient Civilizations exhibit?"
    )
    assert result is not None
    assert result.reasoning_type == "date-difference"
    assert result.answer == "7 days"


def test_extract_question_event_phrases_between_question():
    phrases = extract_question_event_phrases(
        "How many days passed between my visit to the Museum of Modern Art and the Ancient Civilizations exhibit?"
    )
    assert phrases == ["museum modern art", "ancient civilizations exhibit"]


def test_event_phrase_score_prefers_matching_record():
    matching = extract_semantic_state_records(
        "I attended the Ancient Civilizations exhibit at the Metropolitan Museum of Art.",
        date="2024/03/08",
        evidence_id="met:0",
    )[0]
    other = extract_semantic_state_records(
        "I visited the Museum of Modern Art today.",
        date="2024/03/01",
        evidence_id="moma:0",
    )[0]

    assert score_event_phrase("Ancient Civilizations exhibit", matching) > score_event_phrase("Ancient Civilizations exhibit", other)


def test_semantic_state_extractor_captures_travel_and_reward_events():
    records = extract_semantic_state_records(
        "I went on a day hike to Muir Woods. I redeemed $12 cashback for a $10 Amazon gift card.",
        date="2024/03/09",
        evidence_id="events:0",
    )

    values = " ".join(record.value.lower() for record in records)
    assert "day hike" in values
    assert "$12 cashback" in values


def test_state_reasoner_answers_event_order():
    records = [
        *extract_semantic_state_records("I attended Michael's engagement party.", date="2024/01/02", evidence_id="engage:0"),
        *extract_semantic_state_records("I went to my cousin's wedding.", date="2024/02/02", evidence_id="wedding:0"),
    ]

    result = StateReasoner(records).answer("Which event happened first, my cousin's wedding or Michael's engagement party?")
    assert result is not None
    assert result.reasoning_type == "event-order"
    assert "engagement party" in result.answer


def test_state_reasoner_answers_since_reference_date():
    records = extract_semantic_state_records(
        "I attended a friends and family sale at Nordstrom yesterday.",
        date="2024/03/01 (Fri)",
        evidence_id="sale:0",
    )

    result = StateReasoner(records).answer(
        "How many weeks ago did I attend the friends and family sale at Nordstrom?",
        reference_date="2024/03/15 (Fri)",
    )
    assert result is not None
    assert result.reasoning_type == "date-difference"
    assert result.answer == "2"


def test_extract_question_event_phrases_ago_question():
    phrases = extract_question_event_phrases(
        "How many weeks ago did I attend the friends and family sale at Nordstrom?"
    )
    assert phrases == ["attend friends and family sale nordstrom"]


def test_extract_question_event_phrases_quoted_order_question():
    phrases = extract_question_event_phrases(
        "What is the order of the three events: 'I signed up for the rewards program at ShopRite', 'I used a coupon at Walmart', and 'I redeemed cashback from Ibotta'?"
    )
    assert phrases == [
        "signed up for rewards program shoprite",
        "used coupon walmart",
        "redeemed cashback from ibotta",
    ]


def test_semantic_state_extractor_keeps_friends_and_family_sale():
    records = extract_semantic_state_records(
        "I attended a friends and family sale at Nordstrom and picked up a few dresses.",
        date="2024/03/01",
        evidence_id="sale:0",
    )
    assert any("friends and family sale" in record.value.lower() for record in records)


def test_semantic_state_extractor_captures_got_back_from_event():
    records = extract_semantic_state_records(
        "I just got back from a guided tour at the Museum of Modern Art focused on 20th-century modern art movements.",
        date="2024/03/01",
        evidence_id="museum:0",
    )
    assert any("museum of modern art" in record.value.lower() for record in records)


def test_semantic_state_extractor_captures_moved_back_location_state():
    records = extract_state_records(
        "My friend Rachel actually just moved back to the suburbs again.",
        date="2024/03/01",
        evidence_id="move:0",
    )
    assert any(record.record_type == "state" and record.attribute == "location" and "suburbs" in record.value for record in records)


def test_semantic_state_extractor_captures_korean_restaurant_count():
    records = extract_state_records(
        "Have you tried any good Korean restaurants in your city lately? I've tried four different ones so far.",
        date="2024/03/01",
        evidence_id="food:0",
    )
    assert any(record.attribute == "korean restaurants tried count" and record.value == "four" for record in records)


def test_state_reasoner_answers_age_difference():
    records = [
        *extract_state_records("Do you think 32 is considered young or old?", date="2024/02/05", evidence_id="age:0"),
        *extract_state_records("My grandma's 75th birthday celebration was inspiring.", date="2024/02/05", evidence_id="grandma:0"),
    ]

    result = StateReasoner(records).answer("How many years older is my grandma than me?")
    assert result is not None
    assert result.reasoning_type == "age-difference"
    assert result.answer == "43"


def test_state_reasoner_answers_which_event_happened_first_with_wedding_pattern():
    records = [
        *extract_state_records("I just came back from Michael's engagement party at a trendy rooftop bar today.", date="2024/01/01", evidence_id="engage:0"),
        *extract_state_records("I just walked down the aisle as a bridesmaid at my cousin's wedding today.", date="2024/02/01", evidence_id="wedding:0"),
    ]

    result = StateReasoner(records).answer("Which event happened first, my cousin's wedding or Michael's engagement party?")
    assert result is not None
    assert result.answer == "Michael's engagement party"


def test_semantic_event_extractor_infers_explicit_month_day():
    records = extract_state_records(
        'I recently attended a workshop on "Effective Communication in the Workplace" on January 10th.',
        date="2023/01/13 (Fri)",
        evidence_id="workshop:0",
    )

    assert any(record.record_type == "event" and record.date == "2023/01/10" for record in records)


def test_semantic_event_extractor_infers_relative_dates():
    records = extract_state_records(
        "I attended a workshop yesterday. I went on a hike last week. I bought a coffee maker three weeks ago.",
        date="2024/03/15 (Fri)",
        evidence_id="relative:0",
    )
    by_value = {record.value.lower(): record.date for record in records if record.record_type == "event"}

    assert by_value["a workshop yesterday"] == "2024/03/14"
    assert by_value["a hike last week"] == "2024/03/08"
    assert by_value["a coffee maker three weeks ago"] == "2024/02/23"


def test_state_reasoner_uses_harvested_event_for_date_difference():
    records = [
        *extract_state_records(
            "I started watering my herb garden every morning today.",
            date="2023/03/22 (Wed)",
            evidence_id="garden:0",
        ),
        *extract_state_records(
            "I just harvested my first batch of fresh herbs from the herb garden kit today.",
            date="2023/04/15 (Sat)",
            evidence_id="harvest:0",
        ),
    ]

    result = StateReasoner(records).answer(
        "How many days passed between the day I started watering my herb garden and the day I harvested my first batch of fresh herbs?"
    )
    assert result is not None
    assert result.answer == "24 days"


def test_extract_question_event_phrases_before_question():
    phrases = extract_question_event_phrases(
        "How many days before the team meeting I was preparing for did I attend the workshop on 'Effective Communication in the Workplace'?"
    )
    assert phrases == [
        "attend workshop effective communication workplace",
        "team meeting was preparing",
    ]


def test_dated_noun_event_supports_before_question_reasoning():
    records = [
        *extract_state_records(
            'I recently attended a workshop on "Effective Communication in the Workplace" on January 10th.',
            date="2023/01/13 (Fri)",
            evidence_id="workshop:0",
        ),
        *extract_state_records(
            "I remember making a note to myself to practice those skills in my upcoming team meeting on January 17th.",
            date="2023/01/13 (Fri)",
            evidence_id="meeting:0",
        ),
    ]

    result = StateReasoner(records).answer(
        "How many days before the team meeting I was preparing for did I attend the workshop on 'Effective Communication in the Workplace'?"
    )
    assert result is not None
    assert result.answer == "7 days"


def test_state_reasoner_counts_distinct_event_days_in_month():
    records = [
        *extract_state_records(
            "I did a Bible study on this same topic at my church a few weeks ago, on December 17th.",
            date="2024/01/10 (Wed)",
            evidence_id="bible:0",
        ),
        *extract_state_records(
            "I just got back from a lovely midnight mass on Christmas Eve at St. Mary's Church, which was on December 24th, with my family.",
            date="2024/01/10 (Wed)",
            evidence_id="mass:0",
        ),
        *extract_state_records(
            "I helped out at the church's annual holiday food drive on December 10th, sorting donations.",
            date="2024/01/10 (Wed)",
            evidence_id="food:0",
        ),
    ]

    result = StateReasoner(records).answer(
        "How many days did I spend participating in faith-related activities in December?"
    )
    assert result is not None
    assert result.reasoning_type == "distinct-event-day-count"
    assert result.answer == "3"


def test_state_reasoner_sums_explicit_trip_durations():
    records = [
        *extract_state_records(
            "I just got back from a 3-day solo camping trip to Big Sur in early April.",
            date="2023/04/29",
            evidence_id="bigsur:0",
        ),
        *extract_state_records(
            "I just got back from an amazing 5-day camping trip to Yellowstone National Park last month.",
            date="2023/04/29",
            evidence_id="yellowstone:0",
        ),
        *extract_state_records(
            "We had a 7-day family road trip in Utah, but not camping for this time.",
            date="2023/04/29",
            evidence_id="utah:0",
        ),
    ]

    result = StateReasoner(records).answer(
        "How many days did I spend on camping trips in the United States this year?"
    )
    assert result is not None
    assert result.reasoning_type == "duration-sum"
    assert result.answer == "8 days"


def test_latest_state_answers_yoga_frequency_update():
    records = [
        *extract_state_records(
            "I've been doing yoga twice a week, which has really been helping me relax.",
            date="2023/08/11",
            evidence_id="yoga-old:0",
        ),
        *extract_state_records(
            "I've noticed that I'm more focused on days when I attend yoga classes, which is three times a week.",
            date="2023/11/30",
            evidence_id="yoga-new:0",
        ),
    ]

    result = StateReasoner(records).answer("How often do I attend yoga classes to help with my anxiety?")
    assert result is not None
    assert result.answer == "three times a week"


def test_latest_state_answers_bike_count_update():
    records = [
        *extract_state_records("I currently have three bikes, and I'm wondering if that's too many.", date="2023/02/22", evidence_id="bike-old:0"),
        *extract_state_records("I just got a new hybrid bike, so I'll have my road bike, mountain bike, commuter bike, and hybrid bike.", date="2023/10/10", evidence_id="bike-new:0"),
    ]

    result = StateReasoner(records).answer("How many bikes do I currently own?")
    assert result is not None
    assert result.answer == "4"


def test_latest_state_answers_starbucks_gold_stars():
    records = extract_state_records(
        "Actually, I need 120 stars to reach the gold level, not 300.",
        date="2023/07/30",
        evidence_id="stars:0",
    )

    result = StateReasoner(records).answer("How many stars do I need to reach the gold level on my Starbucks Rewards app?")
    assert result is not None
    assert result.answer == "120"


def test_latest_state_answers_named_company_and_lens_location_time():
    records = [
        *extract_state_records("Rachel, an old colleague, who's currently at TechCorp.", date="2023/05/23", evidence_id="company:0"),
        *extract_state_records("I've been getting some great shots with my new 70-200mm zoom lens lately.", date="2023/08/30", evidence_id="lens:0"),
        *extract_state_records("I remember the music shop on Main St where I got my guitar serviced.", date="2023/05/30", evidence_id="guitar:0"),
        *extract_state_records("I'm done with the meeting before I head to the gym, which is usually at 6:00 pm.", date="2023/05/30", evidence_id="gym:0"),
    ]

    assert StateReasoner(records).answer("What company is Rachel currently working at?").answer == "TechCorp"
    assert StateReasoner(records).answer("What type of camera lens did I purchase most recently?").answer == "a 70-200mm zoom lens"
    assert StateReasoner(records).answer("Where did I get my guitar serviced?").answer == "The music shop on Main St."
    assert StateReasoner(records).answer("What time do I usually go to the gym?").answer == "6:00 pm"


def test_previous_latest_state_prefers_older_matching_record():
    records = [
        *extract_state_records(
            "I recently completed a charity 5K run with a personal best time of 27 minutes and 45 seconds.",
            date="2023/04/11",
            evidence_id="run-old:0",
        ),
        *extract_state_records(
            "I finished with a personal best time of 26 minutes and 30 seconds.",
            date="2023/07/30",
            evidence_id="run-new:0",
        ),
    ]

    result = StateReasoner(records).answer("What was my previous personal best time for the charity 5K run?")
    assert result is not None
    assert result.answer == "27 minutes and 45 seconds"


def test_state_reasoner_answers_book_finish_duration():
    records = [
        *extract_state_records(
            'I just started "The Nightingale" by Kristin Hannah today.',
            date="2023/01/10",
            evidence_id="book-start:0",
        ),
        *extract_state_records(
            'I just finished a historical fiction novel, "The Nightingale" by Kristin Hannah, today.',
            date="2023/01/31",
            evidence_id="book-finish:0",
        ),
    ]

    result = StateReasoner(records).answer("How many days did it take me to finish 'The Nightingale' by Kristin Hannah?")
    assert result is not None
    assert result.answer == "21 days"


def test_state_reasoner_answers_since_consecutive_events():
    records = [
        *extract_state_records(
            'I just got back from the "24-Hour Bike Ride" charity event today.',
            date="2023/02/14",
            evidence_id="bike-charity:0",
        ),
        *extract_state_records(
            'I volunteered at the "Books for Kids" charity book drive event at my local library today.',
            date="2023/02/15",
            evidence_id="books-charity:0",
        ),
        *extract_state_records(
            'I just did the "Walk for Hunger" charity event today.',
            date="2023/03/19",
            evidence_id="hunger-charity:0",
        ),
    ]

    result = StateReasoner(records).answer(
        "How many months have passed since I participated in two charity events in a row, on consecutive days?",
        reference_date="2023/04/18",
    )
    assert result is not None
    assert result.reasoning_type == "since-consecutive-events"
    assert result.answer == "2"
