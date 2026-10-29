
from src.agents.data_agent import lookup_faculty


def main():
    result = lookup_faculty("composite materials")

    if result["status"] != "success":
        print(result.get("message", "No matching faculty found."))
        return

    print("Faculty matching composite-material research:\n")

    for faculty in result["results"]:
        print(f"Name: {faculty.get('Name', 'Not provided')}")
        print(f"Department: {faculty.get('Department', 'Not provided')}")
        print(f"Research areas: {faculty.get('Research_Areas', 'Not provided')}")

        profile = faculty.get("Profile_URL", "").strip()
        if profile:
            print(f"Profile: {profile}")

        print()


if __name__ == "__main__":
    main()
