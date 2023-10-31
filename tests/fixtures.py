def user_yes_no(question):
    answer = input(f"{question} Type y or n and press enter.")

    match answer:
        case 'y':
            return 'yes'
        case 'n':
            return 'no'
        case _:
            return 'dunno'
