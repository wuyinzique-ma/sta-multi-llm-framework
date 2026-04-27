class OntologyClass:
    def __init__(self, url: str, name: str=None, label: str=None, comment: str=None):
        """
        Represents a class in the ontology.

        Args:
            url (str): The unique identifier or URL for the ontology class.
            name (str, optional): The name of the ontology
            label (str, optional): The label or name of the ontology class.
            comment (str, optional): A comment or description of the ontology class.
        """
        self.url = url
        self.name = name
        self.label = label
        self.comment = comment

    def __repr__(self):
        return f"OntologyClass(url='{self.url}', name='{self.name}', label='{self.label}', comment='{self.comment}')"