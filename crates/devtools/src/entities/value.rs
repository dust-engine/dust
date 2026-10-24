//! Reflected values, such as a component's fields, as rows of plain data for
//! the inspector to draw. An edit is sent as the new value and the path to it,
//! which [`element_mut`] follows.

use std::any::TypeId;
use std::borrow::Cow;
use std::fmt::{self, Write as _};
use std::time::Duration;

use bevy::ecs::component::ComponentId;
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::reflect::enums::{DynamicEnum, DynamicVariant, Enum, VariantInfo, VariantType};
use bevy::reflect::std_traits::ReflectDefault;
use bevy::reflect::structs::DynamicStruct;
use bevy::reflect::tuple::DynamicTuple;
use bevy::reflect::{PartialReflect, ReflectMut, ReflectRef, TypeRegistry};
use gpui::SharedString;

use super::{Label, WorldInfo};

/// The most elements of a list, map or set shown; the rest are counted.
pub(super) const MAX_ELEMENTS: usize = 100;

/// The value at `path` within `value`. Each index in `path` picks a field of
/// a struct, tuple or enum variant, or an element of a list or array, as
/// [`ValueRows`] records them.
pub(super) fn element_mut<'a>(
    value: &'a mut dyn PartialReflect,
    path: &[usize],
) -> Option<&'a mut dyn PartialReflect> {
    let Some((&index, rest)) = path.split_first() else {
        return Some(value);
    };
    let element = match value.reflect_mut() {
        ReflectMut::Struct(value) => value.field_at_mut(index),
        ReflectMut::TupleStruct(value) => value.field_mut(index),
        ReflectMut::Tuple(value) => value.field_mut(index),
        ReflectMut::List(value) => value.get_mut(index),
        ReflectMut::Array(value) => value.get_mut(index),
        ReflectMut::Enum(value) => value.field_at_mut(index),
        _ => None,
    }?;
    element_mut(element, rest)
}

/// `value` as its `Debug` implementation formats it, on one line and cut at
/// `max` bytes. Formatting stops at the cut, so that previewing a large value
/// costs no more than a small one.
pub(super) fn preview(value: &dyn PartialReflect, max: usize) -> String {
    struct Bounded {
        text: String,
        max: usize,
    }
    impl fmt::Write for Bounded {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            for c in s.chars() {
                if self.text.len() >= self.max {
                    return Err(fmt::Error);
                }
                self.text.push(if c.is_whitespace() { ' ' } else { c });
            }
            Ok(())
        }
    }
    let mut bounded = Bounded {
        text: String::new(),
        max,
    };
    if write!(bounded, "{value:?}").is_err() {
        bounded.text.push('…');
    }
    bounded.text
}

/// A value of `variant` whose fields are their types' defaults, or `None` if
/// one of them doesn't reflect `Default`.
pub(super) fn default_variant(
    variant: &VariantInfo,
    registry: &TypeRegistry,
) -> Option<DynamicEnum> {
    let default = |type_id| {
        let default = registry.get_type_data::<ReflectDefault>(type_id)?;
        Some(default.default().into_partial_reflect())
    };
    let fields = match variant {
        VariantInfo::Unit(_) => DynamicVariant::Unit,
        VariantInfo::Tuple(variant) => {
            let mut fields = DynamicTuple::default();
            for field in variant.iter() {
                fields.insert_boxed(default(field.type_id())?);
            }
            DynamicVariant::Tuple(fields)
        }
        VariantInfo::Struct(variant) => {
            let mut fields = DynamicStruct::default();
            for field in variant.iter() {
                fields.insert_boxed(field.name(), default(field.type_id())?);
            }
            DynamicVariant::Struct(fields)
        }
    };
    Some(DynamicEnum::new(variant.name(), fields))
}

/// [`default_variant`] of the variant named `name` of `value`, an enum.
pub(super) fn named_variant(
    value: &dyn PartialReflect,
    name: &str,
    registry: &TypeRegistry,
) -> Option<DynamicEnum> {
    let ReflectRef::Enum(value) = value.reflect_ref() else {
        return None;
    };
    default_variant(value.get_represented_enum_info()?.variant(name)?, registry)
}

/// Whether [`default_variant`] can build a value of `variant`, without
/// building it.
fn has_default(variant: &VariantInfo, registry: &TypeRegistry) -> bool {
    let default = |type_id| registry.get_type_data::<ReflectDefault>(type_id).is_some();
    match variant {
        VariantInfo::Unit(_) => true,
        VariantInfo::Tuple(variant) => variant.iter().all(|field| default(field.type_id())),
        VariantInfo::Struct(variant) => variant.iter().all(|field| default(field.type_id())),
    }
}

macro_rules! numbers {
    ($($variant:ident($ty:ty)),* $(,)?) => {
        /// A number of one of the primitive types, which the inspector edits
        /// keeping its type.
        #[derive(Clone, Copy, PartialEq, Debug)]
        pub(super) enum Number {
            $($variant($ty),)*
        }

        impl Number {
            fn from_reflect(value: &dyn PartialReflect) -> Option<Self> {
                $(
                    if let Some(&value) = value.try_downcast_ref::<$ty>() {
                        return Some(Number::$variant(value));
                    }
                )*
                None
            }

            /// `text` as a number of the same type.
            pub(super) fn parse(self, text: &str) -> Option<Self> {
                match self {
                    $(Number::$variant(_) => text.trim().parse().ok().map(Number::$variant),)*
                }
            }

            pub(super) fn into_reflect(self) -> Box<dyn PartialReflect> {
                match self {
                    $(Number::$variant(value) => Box::new(value),)*
                }
            }

            pub(super) fn to_f64(self) -> f64 {
                match self {
                    $(Number::$variant(value) => value as f64,)*
                }
            }

            /// `value` as a number of the same type: rounded for an integer,
            /// and clamped to what the type holds.
            pub(super) fn with_f64(self, value: f64) -> Self {
                let value = if self.is_integer() { value.round() } else { value };
                match self {
                    $(Number::$variant(_) => Number::$variant(value as $ty),)*
                }
            }
        }

        impl fmt::Display for Number {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self {
                    $(Number::$variant(value) => value.fmt(f),)*
                }
            }
        }

        /// Whether the fields are two to four numbers, as in `Vec3` or `UVec2`.
        fn is_vector<'a>(fields: impl Iterator<Item = &'a dyn PartialReflect>) -> bool {
            let fields: Vec<&dyn PartialReflect> = fields.collect();
            let numbers = [$(TypeId::of::<$ty>(),)*];
            (2..=4).contains(&fields.len())
                && fields.into_iter().all(|field| {
                    field
                        .get_represented_type_info()
                        .is_some_and(|info| numbers.contains(&info.type_id()))
                })
        }
    };
}

numbers!(
    F32(f32),
    F64(f64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    Isize(isize),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    Usize(usize),
);

impl Number {
    pub(super) fn is_integer(self) -> bool {
        !matches!(self, Number::F32(_) | Number::F64(_))
    }
}

/// A row of a component's value in the inspector.
pub(super) struct ValueRow {
    /// How many collapsible rows the row is nested in.
    pub(super) depth: usize,
    pub(super) kind: ValueRowKind,
}

pub(super) enum ValueRowKind {
    /// A value on one row, after its field's name if it's a field.
    Inline {
        name: Option<SharedString>,
        pieces: Vec<Piece>,
    },
    /// A field whose value doesn't fit on one row. Closed, its header shows a
    /// preview of the value; open, its fields or elements follow, one level
    /// deeper.
    Nested {
        name: SharedString,
        path: Vec<usize>,
        open: bool,
        header: Vec<Piece>,
    },
    /// How many elements weren't sent.
    More(usize),
}

/// A piece of a value's row.
pub(super) enum Piece {
    /// A name within the value, such as the `x` of a `Vec3`.
    Name(SharedString),
    /// What says there is nothing to show, such as `[]`.
    Weak(SharedString),
    /// Text about the value, such as an observer's system.
    Text(SharedString),
    /// A value without a control of its own, as `Debug` formats it.
    Preview(SharedString),
    Leaf(Leaf),
    Variant(Variant),
    Entity(Link),
}

/// A value with a control of its own.
pub(super) struct Leaf {
    /// The path to the value, as [`element_mut`] follows it.
    pub(super) path: Vec<usize>,
    pub(super) value: LeafValue,
    pub(super) editable: bool,
}

pub(super) enum LeafValue {
    Number(Number),
    Bool(bool),
    String(String),
    /// A string the inspector can't edit, such as a `&'static str`, quoted.
    Quoted(SharedString),
    Duration(Duration),
    Rotation(Quat),
}

/// An enum's variant.
pub(super) struct Variant {
    pub(super) path: Vec<usize>,
    pub(super) current: SharedString,
    /// The variants to pick from, each with whether its fields' values can be
    /// built from their types' defaults; `None` if the value isn't editable.
    pub(super) choices: Option<Vec<(SharedString, bool)>>,
}

/// A link to an entity.
pub(super) struct Link {
    pub(super) entity: Entity,
    /// `None` if the entity isn't spawned.
    pub(super) label: Option<Label>,
}

impl Link {
    pub(super) fn new(info: &WorldInfo, entity: Entity) -> Self {
        Link {
            entity,
            label: info.label(entity),
        }
    }
}

/// Builds the rows of a component's value.
pub(super) struct ValueRows<'a, 'w> {
    info: &'a WorldInfo<'w>,
    registry: &'a TypeRegistry,
    /// Whether the value can be edited: `false` within an immutable
    /// component, and within maps and sets, whose elements `element_mut`
    /// can't reach.
    editable: bool,
    /// The component whose value it is, which keys the open rows.
    component: ComponentId,
    /// The paths of the open rows within each component. Keyed by the
    /// component, so that the same fields are open whichever entity shows.
    open: &'a HashSet<(ComponentId, Vec<usize>)>,
    /// Indices from the component down to the value being built, as
    /// `element_mut` follows them.
    path: Vec<usize>,
    depth: usize,
    rows: Vec<ValueRow>,
}

impl<'a, 'w> ValueRows<'a, 'w> {
    pub(super) fn new(
        info: &'a WorldInfo<'w>,
        registry: &'a TypeRegistry,
        editable: bool,
        component: ComponentId,
        open: &'a HashSet<(ComponentId, Vec<usize>)>,
    ) -> Self {
        Self {
            info,
            registry,
            editable,
            component,
            open,
            path: Vec::new(),
            depth: 0,
            rows: Vec::new(),
        }
    }

    pub(super) fn into_rows(self) -> Vec<ValueRow> {
        self.rows
    }

    /// A component's value: on one row if it fits, or else a row per field or
    /// element, after the variant for an enum.
    pub(super) fn show(&mut self, value: &dyn PartialReflect) {
        if self.is_inline(value) {
            let pieces = self.inline(value);
            self.push(ValueRowKind::Inline { name: None, pieces });
            return;
        }
        if let ReflectRef::Enum(value) = value.reflect_ref() {
            let pieces = vec![self.variant(value)];
            self.push(ValueRowKind::Inline { name: None, pieces });
        }
        self.elements(value);
    }

    fn push(&mut self, kind: ValueRowKind) {
        self.rows.push(ValueRow {
            depth: self.depth,
            kind,
        });
    }

    /// Whether `value` fits on one row: a single value, a vector of numbers,
    /// an enum variant with at most one such field, or an empty collection.
    fn is_inline(&self, value: &dyn PartialReflect) -> bool {
        match value.reflect_ref() {
            ReflectRef::Struct(value) => {
                value.field_len() == 0 || is_vector(value.iter_fields().map(|(_, field)| field))
            }
            ReflectRef::TupleStruct(value) => {
                value.field_len() == 0
                    || is_vector(value.iter_fields())
                    || (value.field_len() == 1
                        && value.field(0).is_some_and(|field| self.is_inline(field)))
            }
            ReflectRef::Tuple(value) => value.field_len() == 0 || is_vector(value.iter_fields()),
            ReflectRef::Enum(value) => {
                value.field_len() == 0
                    || (value.variant_type() == VariantType::Tuple
                        && value.field_len() == 1
                        && value.field_at(0).is_some_and(|field| self.is_inline(field)))
            }
            ReflectRef::List(value) => value.is_empty(),
            ReflectRef::Array(value) => value.is_empty(),
            ReflectRef::Map(value) => value.is_empty(),
            ReflectRef::Set(value) => value.is_empty(),
            _ => true,
        }
    }

    /// The row of a field: its name, then its value on the same row if
    /// [`Self::is_inline`], or else below it.
    fn field(&mut self, name: SharedString, value: &dyn PartialReflect) {
        if self.is_inline(value) {
            let pieces = self.inline(value);
            self.push(ValueRowKind::Inline {
                name: Some(name),
                pieces,
            });
        } else {
            self.nested(name, value);
        }
    }

    /// A collapsible row for a value that doesn't fit on one: closed, it
    /// shows a preview of the value; open, its fields or elements below.
    fn nested(&mut self, name: SharedString, value: &dyn PartialReflect) {
        let open = self.open.contains(&(self.component, self.path.clone()));
        let header = match value.reflect_ref() {
            // The variant can be picked without opening the row.
            ReflectRef::Enum(value) => vec![self.variant(value)],
            reflected => {
                let count = match reflected {
                    ReflectRef::List(value) => Some(value.len()),
                    ReflectRef::Array(value) => Some(value.len()),
                    ReflectRef::Map(value) => Some(value.len()),
                    ReflectRef::Set(value) => Some(value.len()),
                    _ => None,
                };
                let mut header = Vec::new();
                if let Some(count) = count {
                    header.push(Piece::Weak(format!("({count})").into()));
                }
                if !open {
                    header.push(Piece::Preview(preview(value, 200).into()));
                }
                header
            }
        };
        self.push(ValueRowKind::Nested {
            name,
            path: self.path.clone(),
            open,
            header,
        });
        if open {
            self.depth += 1;
            self.elements(value);
            self.depth -= 1;
        }
    }

    /// A row per field or element of `value`.
    fn elements(&mut self, value: &dyn PartialReflect) {
        let mut more = 0;
        match value.reflect_ref() {
            ReflectRef::Struct(value) => {
                for (index, (name, element)) in value.iter_fields().enumerate() {
                    self.element(index, SharedString::new(name), element);
                }
            }
            ReflectRef::TupleStruct(value) => {
                for (index, element) in value.iter_fields().enumerate() {
                    self.element(index, index.to_string().into(), element);
                }
            }
            ReflectRef::Tuple(value) => {
                for (index, element) in value.iter_fields().enumerate() {
                    self.element(index, index.to_string().into(), element);
                }
            }
            ReflectRef::Enum(value) => {
                for (index, element) in value.iter_fields().enumerate() {
                    let name = match element.name() {
                        Some(name) => SharedString::new(name),
                        None => index.to_string().into(),
                    };
                    self.element(index, name, element.value());
                }
            }
            ReflectRef::List(value) => {
                for (index, element) in value.iter().take(MAX_ELEMENTS).enumerate() {
                    self.element(index, index.to_string().into(), element);
                }
                more = value.len().saturating_sub(MAX_ELEMENTS);
            }
            ReflectRef::Array(value) => {
                for (index, element) in value.iter().take(MAX_ELEMENTS).enumerate() {
                    self.element(index, index.to_string().into(), element);
                }
                more = value.len().saturating_sub(MAX_ELEMENTS);
            }
            ReflectRef::Map(value) => {
                let editable = std::mem::replace(&mut self.editable, false);
                for (index, (key, element)) in value.iter().take(MAX_ELEMENTS).enumerate() {
                    self.element(index, preview(key, 40).into(), element);
                }
                self.editable = editable;
                more = value.len().saturating_sub(MAX_ELEMENTS);
            }
            ReflectRef::Set(value) => {
                let editable = std::mem::replace(&mut self.editable, false);
                for (index, element) in value.iter().take(MAX_ELEMENTS).enumerate() {
                    self.element(index, index.to_string().into(), element);
                }
                self.editable = editable;
                more = value.len().saturating_sub(MAX_ELEMENTS);
            }
            _ => {
                let pieces = self.inline(value);
                self.push(ValueRowKind::Inline { name: None, pieces });
            }
        }
        if more > 0 {
            self.push(ValueRowKind::More(more));
        }
    }

    /// The row of the field or element at `index` of the value being built.
    fn element(&mut self, index: usize, name: SharedString, value: &dyn PartialReflect) {
        self.path.push(index);
        self.field(name, value);
        self.path.pop();
    }

    /// `value` on one row.
    fn inline(&mut self, value: &dyn PartialReflect) -> Vec<Piece> {
        let mut pieces = Vec::new();
        self.inline_into(value, &mut pieces);
        pieces
    }

    fn inline_into(&mut self, value: &dyn PartialReflect, pieces: &mut Vec<Piece>) {
        if let Some(leaf) = self.leaf(value) {
            pieces.push(leaf);
            return;
        }
        match value.reflect_ref() {
            ReflectRef::Struct(fields) if fields.field_len() > 0 => {
                for (index, (name, field)) in fields.iter_fields().enumerate() {
                    pieces.push(Piece::Name(SharedString::new(name)));
                    self.path.push(index);
                    self.inline_into(field, pieces);
                    self.path.pop();
                }
            }
            ReflectRef::TupleStruct(fields) if fields.field_len() == 1 => {
                // A newtype shows as the value it wraps.
                self.path.push(0);
                self.inline_into(fields.field(0).expect("one field"), pieces);
                self.path.pop();
            }
            ReflectRef::TupleStruct(fields) if fields.field_len() > 0 => {
                for (index, field) in fields.iter_fields().enumerate() {
                    pieces.push(Piece::Name(index.to_string().into()));
                    self.path.push(index);
                    self.inline_into(field, pieces);
                    self.path.pop();
                }
            }
            ReflectRef::Tuple(fields) if fields.field_len() > 0 => {
                for (index, field) in fields.iter_fields().enumerate() {
                    pieces.push(Piece::Name(index.to_string().into()));
                    self.path.push(index);
                    self.inline_into(field, pieces);
                    self.path.pop();
                }
            }
            ReflectRef::Enum(value) => {
                pieces.push(self.variant(value));
                if let Some(field) = value.field_at(0) {
                    self.path.push(0);
                    self.inline_into(field, pieces);
                    self.path.pop();
                }
            }
            ReflectRef::List(_) | ReflectRef::Array(_) => {
                pieces.push(Piece::Weak(SharedString::new_static("[]")));
            }
            ReflectRef::Map(_) | ReflectRef::Set(_) => {
                pieces.push(Piece::Weak(SharedString::new_static("{}")));
            }
            ReflectRef::Struct(_) | ReflectRef::TupleStruct(_) | ReflectRef::Tuple(_) => {
                // A unit struct; its type is all there is to it.
                pieces.push(Piece::Weak(
                    value.reflect_short_type_path().to_owned().into(),
                ));
            }
            _ => {
                // An opaque type without a control of its own.
                pieces.push(Piece::Preview(preview(value, 200).into()));
            }
        }
    }

    /// `value` if it's of a type with a control of its own.
    fn leaf(&self, value: &dyn PartialReflect) -> Option<Piece> {
        let leaf = |value, editable| {
            Some(Piece::Leaf(Leaf {
                path: self.path.clone(),
                value,
                editable,
            }))
        };
        if let Some(number) = Number::from_reflect(value) {
            leaf(LeafValue::Number(number), self.editable)
        } else if let Some(&value) = value.try_downcast_ref::<bool>() {
            leaf(LeafValue::Bool(value), self.editable)
        } else if let Some(value) = value.try_downcast_ref::<String>() {
            leaf(LeafValue::String(value.clone()), self.editable)
        } else if let Some(value) = value
            .try_downcast_ref::<Cow<'static, str>>()
            .map(|value| &**value)
            .or_else(|| value.try_downcast_ref::<&'static str>().copied())
        {
            leaf(LeafValue::Quoted(format!("{value:?}").into()), false)
        } else if let Some(value) = value.try_downcast_ref::<char>() {
            leaf(LeafValue::Quoted(format!("{value:?}").into()), false)
        } else if let Some(&entity) = value.try_downcast_ref::<Entity>() {
            Some(Piece::Entity(Link::new(self.info, entity)))
        } else if let Some(&duration) = value.try_downcast_ref::<Duration>() {
            leaf(LeafValue::Duration(duration), self.editable)
        } else if let Some(&rotation) = value.try_downcast_ref::<Quat>() {
            leaf(LeafValue::Rotation(rotation), self.editable)
        } else {
            None
        }
    }

    /// The enum's variant, with the variants to pick from when editable.
    fn variant(&self, value: &dyn Enum) -> Piece {
        let choices = value
            .get_represented_enum_info()
            .filter(|_| self.editable)
            .map(|info| {
                info.iter()
                    .map(|variant| {
                        let name = SharedString::new_static(variant.name());
                        (name, has_default(variant, self.registry))
                    })
                    .collect()
            });
        Piece::Variant(Variant {
            path: self.path.clone(),
            current: value.variant_name().to_owned().into(),
            choices,
        })
    }
}
